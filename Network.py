import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    通道混洗（Channel Shuffle）：解决分组卷积组间信息不流通的问题
    Args:
        x: 输入特征图 (batch_size, channels, height, width)
        groups: 分组数
    Returns:
        混洗后的特征图
    """
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # 步骤1：将通道维度重塑为 (groups, channels_per_group)
    x = x.view(batch_size, groups, channels_per_group, height, width)
    # 步骤2：转置交换groups和channels_per_group的维度
    x = torch.transpose(x, 1, 2).contiguous()
    # 步骤3：展平回原始通道维度
    x = x.view(batch_size, -1, height, width)
    return x


class ShuffleNetUnit(nn.Module):
    """
    ShuffleNet基本单元（包含两种结构：stride=1的普通单元和stride=2的降采样单元）
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int, groups: int):
        super().__init__()
        self.stride = stride
        self.groups = groups
        bottleneck_channels = out_channels // 4  # 瓶颈层通道数为输出的1/4

        # 1x1分组卷积（GConv）
        self.gconv1 = nn.Conv2d(
            in_channels, bottleneck_channels, kernel_size=1, groups=groups, bias=False
        )
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        # 3x3深度可分离卷积（DWConv）
        self.dwconv = nn.Conv2d(
            bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride,
            padding=1, groups=bottleneck_channels, bias=False
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        # 1x1分组卷积（GConv）
        # 若stride=2，输出通道需减去shortcut的in_channels（因为后续会concat）
        gconv2_out_channels = out_channels - (in_channels if stride == 2 else 0)
        self.gconv2 = nn.Conv2d(
            bottleneck_channels, gconv2_out_channels, kernel_size=1, groups=groups, bias=False
        )
        self.bn3 = nn.BatchNorm2d(gconv2_out_channels)

        # Shortcut分支：stride=2时用平均池化降采样，否则直接恒等映射
        self.shortcut = nn.AvgPool2d(kernel_size=3, stride=2, padding=1) if stride == 2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        # 主路径：1x1 GConv -> BN -> ReLU -> Channel Shuffle -> 3x3 DWConv -> BN -> 1x1 GConv -> BN
        out = self.gconv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = channel_shuffle(out, self.groups)
        
        out = self.dwconv(out)
        out = self.bn2(out)
        
        out = self.gconv2(out)
        out = self.bn3(out)

        # 融合：stride=2时concat，否则add
        if self.stride == 2:
            out = torch.cat([out, residual], dim=1)
        else:
            out = out + residual

        return F.relu(out)

class ShuffleNet(nn.Module):
    """
    ShuffleNet V1 完整网络架构
    Args:
        num_classes: 分类类别数（默认ImageNet 1000类）
        groups: 分组数（支持1/2/3/4/8，原文默认3）
        scale_factor: 通道缩放因子（如0.5x/1.0x/1.5x/2.0x，原文默认1.0x）
    """
    def __init__(self, num_classes: int = 1000, groups: int = 3, scale_factor: float = 1.0):
        super().__init__()
        self.groups = groups

        # 定义不同分组数对应的各阶段输出通道数（原文Table 1）
        group_to_channels = {
            1: [144, 288, 576],
            2: [200, 400, 800],
            3: [240, 480, 960],
            4: [272, 544, 1088],
            8: [384, 768, 1536]
        }
        if groups not in group_to_channels:
            raise ValueError(f"Groups must be one of {list(group_to_channels.keys())}")
        
        # 应用通道缩放因子
        stage_out_channels = [int(c * scale_factor) for c in group_to_channels[groups]]

        # 初始层：Conv1 -> BN -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 构建Stage 2-4（每个Stage包含多个ShuffleNet Unit）
        self.stage2 = self._make_stage(24, stage_out_channels[0], num_units=3, stride=2)
        self.stage3 = self._make_stage(stage_out_channels[0], stage_out_channels[1], num_units=7, stride=2)
        self.stage4 = self._make_stage(stage_out_channels[1], stage_out_channels[2], num_units=3, stride=2)

        # 分类头：全局平均池化 -> 全连接层
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage_out_channels[2], num_classes)

        # 权重初始化
        self._initialize_weights()

    def _make_stage(self, in_channels: int, out_channels: int, num_units: int, stride: int) -> nn.Sequential:
        """构建单个Stage（第一个Unit降采样，其余Unit保持分辨率）"""
        units = [ShuffleNetUnit(in_channels, out_channels, stride=stride, groups=self.groups)]
        for _ in range(num_units - 1):
            units.append(ShuffleNetUnit(out_channels, out_channels, stride=1, groups=self.groups))
        return nn.Sequential(*units)

    def _initialize_weights(self):
        """Kaiming初始化卷积层权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def test_shufflenet():
    # 初始化模型（以groups=3, scale=1.0x为例）
    model = ShuffleNet(num_classes=1000, groups=3, scale_factor=1.0)
    print(model)  # 打印网络结构

    # 测试前向传播（输入224x224的ImageNet格式图片）
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\n输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    assert output.shape == (1, 1000), "输出形状不匹配！"
    print("测试通过！")


# 运行测试
test_shufflenet()
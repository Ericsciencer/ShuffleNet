import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# ----------------------
# 1. 核心组件：通道混洗（独立函数，方便复用）
# ----------------------
def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x


# ----------------------
# 2. ShuffleNet 基本单元
# ----------------------
class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups):
        super().__init__()
        self.stride = stride
        self.groups = groups
        bottleneck_channels = out_channels // 4

        # 1x1 分组卷积
        self.gconv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        # 3x3 深度可分离卷积
        self.dwconv = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, stride, 1, groups=bottleneck_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        # 1x1 分组卷积（输出通道适配 shortcut）
        gconv2_out_channels = out_channels - (in_channels if stride == 2 else 0)
        self.gconv2 = nn.Conv2d(bottleneck_channels, gconv2_out_channels, 1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(gconv2_out_channels)
        # Shortcut 分支
        self.shortcut = nn.AvgPool2d(3, 2, 1) if stride == 2 else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.gconv1(x)))
        out = channel_shuffle(out, self.groups)
        out = self.bn2(self.dwconv(out))
        out = self.bn3(self.gconv2(out))
        out = torch.cat([out, residual], dim=1) if self.stride == 2 else out + residual
        return F.relu(out)


# ----------------------
# 3. 适配 CIFAR-10 的 ShuffleNet 完整模型
# ----------------------
class ShuffleNet_CIFAR10(nn.Module):
    def __init__(self, num_classes=10, groups=3, scale_factor=0.5):
        super().__init__()
        self.groups = groups
        # 不同分组数对应的通道配置（缩小0.5x适配小数据集）
        group_to_channels = {1: [144, 288, 576], 2: [200, 400, 800], 3: [240, 480, 960]}
        stage_out_channels = [int(c * scale_factor) for c in group_to_channels[groups]]

        # 初始层（步长改为1，适配32x32输入）
        self.conv1 = nn.Conv2d(3, 24, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        # Stage 2-4
        self.stage2 = self._make_stage(24, stage_out_channels[0], 3, stride=2)
        self.stage3 = self._make_stage(stage_out_channels[0], stage_out_channels[1], 7, stride=2)
        self.stage4 = self._make_stage(stage_out_channels[1], stage_out_channels[2], 3, stride=2)
        # 分类头
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage_out_channels[2], num_classes)
        self._initialize_weights()

    def _make_stage(self, in_channels, out_channels, num_units, stride):
        units = [ShuffleNetUnit(in_channels, out_channels, stride, self.groups)]
        units += [ShuffleNetUnit(out_channels, out_channels, 1, self.groups) for _ in range(num_units - 1)]
        return nn.Sequential(*units)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d): nn.init.constant_(m.weight, 1), nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear): nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ----------------------
# 4. 数据加载与预处理
# ----------------------
def get_data_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


# ----------------------
# 5. 训练函数
# ----------------------
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += loss.item() * images.size(0)
    avg_train_loss = total_loss / len(train_loader.dataset)
    avg_train_acc = correct / total
    return avg_train_loss, avg_train_acc


# ----------------------
# 6. 测试函数
# ----------------------
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# ----------------------
# 7. 主程序（含可视化）
# ----------------------
if __name__ == '__main__':
    # 基础配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 128
    lr = 0.1
    num_epochs = 10

    # 初始化核心组件
    model = ShuffleNet_CIFAR10(num_classes=10, groups=3, scale_factor=0.5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=4e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    train_loader, test_loader = get_data_loaders(batch_size)

    # 存储指标的列表
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # 训练循环
    print(f"Training on {device}...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_acc = test(model, test_loader, device)
        scheduler.step()
        
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), 'shufflenet_cifar10.pth')
    print("Model saved as shufflenet_cifar10.pth")

    # 可视化绘图
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 7))
    plt.plot(epochs, train_loss_list, 'b-', linewidth=2, label='train loss')
    plt.plot(epochs, train_acc_list, 'm--', linewidth=2, label='train acc')
    plt.plot(epochs, test_acc_list, 'g--', linewidth=2, label='test acc')
    plt.xlabel('epoch', fontsize=18)
    plt.xticks(range(2, 11, 2))
    plt.ylim(0, 2.4)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=18)
    plt.title('ShuffleNet Training Metrics', fontsize=16)
    plt.savefig('shufflenet_training_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
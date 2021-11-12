# 创建ResNet50模型,共5个stage（stage1 --> stage5，即论文中的conv1--> conv5）

import torch
from torch import nn


# 定义初始卷积层，即stage1
def stage_init(channel_in, channel_out, stride=(2, 2)):
    return nn.Sequential(
        # 卷积：卷积核=7*7，stride=2，padding=3
        nn.Conv2d(in_channels=channel_in, out_channels=channel_out,
                  kernel_size=(7, 7), stride=stride, padding=3, bias=False),
        # BN归一化
        nn.BatchNorm2d(channel_out),
        # ReLU激活
        nn.ReLU(inplace=True),  # inplace=true时不创建新的对象，直接对原始对象进行修改，节省内存
        # 池化：kernel=(3, 3), stride=2, padding=1
        nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
    )


# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, channel_in, channel_out, stride=(1, 1), down_sample=False, expansion=4):
        # 输入输出通道数为第一个卷积层输入输出通道数
        super(ResidualBlock, self).__init__()
        self.down_sample = down_sample  # 是否进行降采样，以确定残差传递方式
        self.expansion = expansion  # 通道放大倍数，输出通道数=输入通道数*expansion
        # 定义三重卷积叠加，进行残差传递
        self.residual_block = nn.Sequential(
            # 1*1卷积，kernel=1，stride=1，padding默认为0（保证输入输出尺寸大小一致），输入输出通道不一致
            nn.Conv2d(in_channels=channel_in, out_channels=channel_out,
                      kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace=True),

            # 3*3卷积，kernel=3，stride=stride(stage1中为1，其余stage中为2)，
            # padding=1（stride=1尺寸不变，stride=2尺寸减半）,输入输出通道一致
            nn.Conv2d(in_channels=channel_out, out_channels=channel_out,
                      kernel_size=(3, 3), stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace=True),

            # 1*1卷积，kernel=1，stride=1，padding为默认0（输入输出尺寸大小一致），输入输出通道不一致
            nn.Conv2d(in_channels=channel_out, out_channels=channel_out*self.expansion,
                      kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(channel_out*self.expansion),
        )

        # 是否降采样，是则进行1*1卷积，不是则直接传递，保证残差传递输入输出维度一致
        if self.down_sample:
            # 设定降采样函数
            self.ds = nn.Sequential(
                nn.Conv2d(in_channels=channel_in, out_channels=channel_out*self.expansion,
                          kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(channel_out*self.expansion)
            )

        self.relu = nn.ReLU(inplace=True)

    # 定义残差块前向传播函数
    def forward(self, input_x):
        residual = input_x  # 定义初始残差（论文中的x），即输入数据
        output_x = self.residual_block(input_x)  # 计算残差块(论文中的F(X))

        # 输入输出不同维度则进行降采样，执行if语句；同维度则跳过if语句，直接向后传递residual
        if self.down_sample:
            residual = self.ds(input_x)  # 进行降采样时，对输入进行1*1卷积

        # 计算 H(x) = F(x) + x
        output_x += residual
        output_x = self.relu(output_x)
        return output_x


# 定义整体网络模型ResNet50
class MyModel(nn.Module):
    def __init__(self, blocks=None, num_classes=1000, expansion=4):
        # blocks为各个stage中残差块数量，有四个含残差块stage，对应一个四维列表
        # num_classes为输出类别数量，针对ImageNet默认1000类
        super(MyModel, self).__init__()
        # expansion输出相对输入的通道数量扩展倍数，四个stage均为默认4
        if blocks is None:
            blocks = [3, 4, 6, 3]  # 默认resnet50各stage残差块数为 3，4，6，3
        self.expansion = expansion

        # 调用stage_init，网络第一个大块
        self.stage1 = stage_init(channel_in=3, channel_out=64)

        # 调用make_stage完成后续四大块stage2~5
        self.stage2 = self.make_stage(channel_in=64, channel_out=64, stride=1, block=blocks[0])
        # 只有stage2的stride全为1，其余stage在3*3卷积和降采样中stride为2
        self.stage3 = self.make_stage(channel_in=256, channel_out=128, stride=2, block=blocks[1])
        self.stage4 = self.make_stage(channel_in=512, channel_out=256, stride=2, block=blocks[2])
        self.stage5 = self.make_stage(channel_in=1024, channel_out=512, stride=2, block=blocks[3])

        # 平均池化与全连接层
        self.average_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(in_features=2048, out_features=num_classes)
        # 2048 = 512 * 4 即 channel_out*expansion
        # 初始化卷积层与BN层参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_stage(self, channel_in, channel_out, stride, block):
        # 含残差计算的四大块由make_stage完成
        # 输入输出通道数为stage第一个残差块的第一个卷积层输入输出通道数，block为单个stage残差块数
        stages = [ResidualBlock(channel_in, channel_out, stride, down_sample=True)]
        # 每个stage的第一个残差块都进行降采样，down_sample都设为true
        # stage中的剩余残差块，循环从第二个残差块开始，剩余残差块中stride均为1默认值
        for i in range(1, block):
            stages.append(ResidualBlock(channel_out*self.expansion, channel_out))

        return nn.Sequential(*stages)  # 展开stages，顺序完成各个残差块

    def forward(self, x):
        # 定义前向传播函数
        x = self.stage1(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        x = self.average_pool(x)
        x = x.view(x.size(0), -1)
        # 将前面操作输出的多维度的tensor展平成一维，然后输入分类器，类似于nn.Flatten()
        # -1是自适应分配，指在不知道函数有多少列的情况下，根据原tensor数据自动分配列数。
        x = self.fc(x)

        return x


if __name__ == '__main__':
    model = MyModel()
    print(model)

    in_put = torch.randn(1, 3, 224, 224)
    out = model(in_put)
    print(out.shape)

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init


class MySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = input.new(input.size())
        output[input >= 0] = 0.01
        output[input < 0] = -0.01
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class Depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in=64, ch_out=64):
        super(Depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


# CA
class ChannelAttention(nn.Module):
    def __init__(self, in_planes=64, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels=in_planes, out_channels=in_planes // ratio, kernel_size=(1, 1), bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels=in_planes // ratio, out_channels=in_planes, kernel_size=(1, 1), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# SA
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()

        assert kernel_size in ((3, 3), (7, 7)), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == (7, 7) else 1

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding,
                               bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# AM
class AM(nn.Module):
    def __init__(self, in_planes=64, ratio=16, kernel_size=(7, 7)):
        super(AM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


# CBAM
class CBAM(nn.Module):
    def __init__(self):
        super(CBAM, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            AM(),
            nn.Conv2d(64, 1, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        res1 = self.layers(x)
        return res1


class StudentNet(nn.Module):
    def __init__(self, num_features, num_blocks):
        super(StudentNet, self).__init__()
        self.num_blocks = num_blocks
        self.num_features = num_features
        # 正交二元矩阵
        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(num_features, 1024)))
        # 测量矩阵
        self.MyBinarize = MySign.apply
        # 初始重构
        self.initialization = torch.nn.Sequential(
            nn.ConvTranspose2d(in_channels=num_features, out_channels=1, kernel_size=(32, 32), stride=(32, 32),
                               padding=(0, 0), bias=True)
        )

        # shallow feature extraction
        self.getFactor = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.CBAMs = nn.ModuleList([CBAM()])
        for _ in range(self.num_blocks - 1):
            self.CBAMs.append(CBAM())

        self.gff = nn.Sequential(
            nn.Conv2d(64 * self.num_blocks, 64, kernel_size=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        PhiWeight = self.MyBinarize(self.Phi).contiguous().view(self.num_features, 1, 32, 32)
        sample_number = F.conv2d(x, PhiWeight, padding=0, stride=32, bias=None)  # Get measurements
        initial_number = self.initialization(sample_number)
        factor = self.getFactor(initial_number)
        x = factor

        local_features = []
        for i in range(self.num_blocks):
            x = self.CBAMs[i](x)
            local_features.append(x)

        out = self.gff(torch.cat(local_features, 1)) + initial_number
        return out

    def get_first_out(self, x):
        PhiWeight = self.MyBinarize(self.Phi).contiguous().view(self.num_features, 1, 32, 32)
        sample_number = F.conv2d(x, PhiWeight, padding=0, stride=32, bias=None)  # Get measurements
        initial_number = self.initialization(sample_number)
        factor = self.getFactor(initial_number)
        for i in range(1):
            factor = self.CBAMs[i](factor)
        return factor

    def get_second_out(self, x):
        PhiWeight = self.MyBinarize(self.Phi).contiguous().view(self.num_features, 1, 32, 32)
        sample_number = F.conv2d(x, PhiWeight, padding=0, stride=32, bias=None)  # Get measurements
        initial_number = self.initialization(sample_number)
        factor = self.getFactor(initial_number)
        for i in range(2):
            factor = self.CBAMs[i](factor)
        return factor

    def get_third_out(self, x):
        PhiWeight = self.MyBinarize(self.Phi).contiguous().view(self.num_features, 1, 32, 32)
        sample_number = F.conv2d(x, PhiWeight, padding=0, stride=32, bias=None)  # Get measurements
        initial_number = self.initialization(sample_number)
        factor = self.getFactor(initial_number)
        for i in range(3):
            factor = self.CBAMs[i](factor)
        return factor

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
    elif isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0.1)


if __name__ == '__main__':
    import time
    from torchsummary import summary

    start_time = time.time()
    model = StudentNet(num_features=102, num_blocks=3).cuda()

    # print(model.MyBinarize())
    summary(model, input_size=(1, 32, 32))
    end_time = time.time()
    print("total_time:", end_time - start_time)

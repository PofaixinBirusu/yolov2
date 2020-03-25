import torch
from torch import nn
from torch.nn import functional as F


class PassThrough(nn.Module):
    def __init__(self):
        super(PassThrough, self).__init__()

    def forward(self, x):
        batch_size, channel, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        return torch.cat([
            x[:, :, ::2, :].contiguous()
                .view(batch_size, channel, -1, 2)
                .permute([0, 1, 3, 2]).contiguous()
                .view(batch_size, -1, h//2, w//2),
            x[:, :, 1::2, :].contiguous()
                .view(batch_size, channel, -1, 2)
                .permute([0, 1, 3, 2]).contiguous()
                .view(batch_size, -1, h//2, w//2)
        ], dim=1)


class Darknet19(nn.Module):
    def __init__(self, n_class, in_channel=3, anchor_num=5):
        super(Darknet19, self).__init__()
        current_channel = in_channel
        self.passthrough = PassThrough()
        self.n_class = n_class
        Conv3x3, Conv1x1, Maxpooling, Pazzthrough, Cat = "Conv3x3", "Conv1x1", "MaxPooling", "Passthrough", "Concat"
        layers = [
            # kernel_size, padding, out_channel
            [Conv3x3, 32], Maxpooling,
            [Conv3x3, 64], Maxpooling,
            [Conv3x3, 128], [Conv1x1, 64], [Conv3x3, 128], Maxpooling,
            [Conv3x3, 256], [Conv1x1, 128], [Conv3x3, 256], Maxpooling,
            [Conv3x3, 512], [Conv1x1, 256], [Conv3x3, 512], [Conv1x1, 256], [Conv3x3, 512],
            Pazzthrough,
            Maxpooling, [Conv3x3, 1024], [Conv1x1, 512], [Conv3x3, 1024], [Conv1x1, 512], [Conv3x3, 1024],
            [Conv3x3, 1024], #[Conv3x3, 1024],
            Cat,
            [Conv3x3, 1024],
        ]
        before_passthrough, after_passthrough, self.after_cat = [], [], None
        before = True
        for layer in layers:
            layer_list = before_passthrough if before else after_passthrough
            if layer[0] == Conv3x3:
                layer_list.append(self.conv(current_channel, layer[1], kernel_size=3, padding=1))
                current_channel = layer[1]
            elif layer[0] == Conv1x1:
                layer_list.append(self.conv(current_channel, layer[1], kernel_size=1, padding=0))
                current_channel = layer[1]
            elif layer[0] == Maxpooling[0]:
                layer_list.append(nn.MaxPool2d(2, 2))
            elif layer[0] == Pazzthrough[0]:
                before = False
            elif layer[0] == Cat[0]:
                break
        self.after_cat = self.conv(current_channel*3, layers[-1][1], 3, 1)
        self.before_passthrough = nn.Sequential(*before_passthrough)
        self.after_passthrough = nn.Sequential(*after_passthrough)
        self.fc = nn.Conv2d(layers[-1][1], anchor_num*(5+n_class), kernel_size=1, stride=1, bias=False)

    def conv(self, in_channel, out_channel, kernel_size, padding, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channel, momentum=0.01),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        before_passthrough = self.before_passthrough(x)
        after_passthrough = self.after_passthrough(before_passthrough)
        passthrough_concat = torch.cat([self.passthrough(before_passthrough), after_passthrough], dim=1)
        out = self.after_cat(passthrough_concat)
        out = self.fc(out)
        out = out.permute([0, 2, 3, 1]).contiguous()\
            .view(x.shape[0], -1, 5*(5+self.n_class))\
            .view(x.shape[0], -1, 5+self.n_class)
        xy_pred = torch.sigmoid(out[:, :, 0:2])
        wh_pred = out[:, :, 2:4]
        # bs*wh5*_
        confidence_pred = torch.sigmoid(out[:, :, 4:5])
        class_pred = torch.sigmoid(out[:, :, 5:])
        box_pred = torch.cat([xy_pred, wh_pred], dim=2)
        return box_pred, confidence_pred, class_pred


if __name__ == '__main__':
    # x = [[1,   2,  3,  4,  5,  6],
    #      [7,   8,  9, 10, 11, 12],
    #      [13, 14, 15, 16, 17, 18],
    #      [19, 20, 21, 22, 23, 24],
    #      [25, 26, 27, 28, 29, 30],
    #      [31, 32, 33, 34, 35, 36]]
    # x = torch.Tensor(x).view(1, 1, 6, 6)
    # passthrough = PassThrough()
    # print(passthrough(x), passthrough(x).shape)
    # y = torch.arange(1, 17).view(1, 1, 4, 4)
    # print(y)
    # print(passthrough(y))
    x = torch.rand(5, 3, 416, 416).cuda()
    net = Darknet19(n_class=20).cuda()
    print(net(x).shape)

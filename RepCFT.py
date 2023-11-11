import torch.nn as nn
import numpy as np
import torch
import copy
import torch.utils.checkpoint as checkpoint

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))

    return result

def avg_bn(out_channels, kernel_size, stride, padding):
    layer = nn.Sequential()
    layer.add_module('avg', nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
    layer.add_module('avgbn', nn.BatchNorm2d(num_features=out_channels))

    return layer

def transI_fusebn(kernel, bn):
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std

def transIII_1x1_kxk(k1, b1, k2, b2, groups):
    assert groups == 1
    k = nn.functional.conv2d(k2, k1.permute(1, 0, 2, 3))      #
    b_hat = (k2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3))
    return k, b_hat + b2

def transV_avg(channels, kernel_size, groups):
    input_dim = channels // groups
    k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
    k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
    return k

class RepLGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(RepLGBlock, self).__init__()
        self.deploy = deploy
        self.groups = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        # self.nolinearity = nn.ReLU6()
        self.nonlinearity = nn.ReLU()

        if deploy:
            self.rbr_reparam_dw = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride,
                                         padding=3 // 2, dilation=dilation, groups=in_channels, bias=True, padding_mode=padding_mode)

            self.rbr_reparam_pw = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                         padding=3 // 2, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.rbr_3x3_dw = conv_bn(in_channels, in_channels, kernel_size, stride, padding=kernel_size // 2, groups=in_channels)
            self.rbr_1x3_dw = conv_bn(in_channels, in_channels,  kernel_size=(1, 3), stride=stride, padding=(0, 1), groups=in_channels)
            self.rbr_3x1_dw = conv_bn(in_channels, in_channels,  kernel_size=(3, 1), stride=stride, padding=(1, 0), groups=in_channels)
            self.rbr_id = nn.BatchNorm2d(in_channels) if stride == 1 else None

            self.rbr_pw = conv_bn(in_channels, out_channels, stride=1, kernel_size=1, padding=0)

            self.rbr_avg = avg_bn(out_channels, 3, 1, 1)

    def forward(self, inp):
        if hasattr(self, 'rbr_reparam_dw'):
            return self.rbr_reparam_pw(self.nonlinearity(self.rbr_reparam_dw(inp)))

        if self.rbr_id is None:
            id_out = 0
        else:
            id_out = self.rbr_id(inp)

        return self.rbr_avg(self.rbr_pw(self.nonlinearity(self.rbr_3x3_dw(inp) + self.rbr_1x3_dw(inp) + self.rbr_3x1_dw(inp) + id_out)))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_3x3_dw)
        kernel1x3, bias1x3 = self._fuse_bn_tensor(self.rbr_1x3_dw)
        kernel3x1, bias3x1 = self._fuse_bn_tensor(self.rbr_3x1_dw)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_id)

        pw1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_pw)

        k_avg = transV_avg(self.out_channels, 3, 1)
        kernelavg, biasavg = transI_fusebn(k_avg.to(self.rbr_avg.avgbn.weight.device), self.rbr_avg.avgbn)

        pw_kernel_merged, pw_bias_merged =  transIII_1x1_kxk(pw1x1, bias1x1, kernelavg, biasavg, 1)



        kerneldw = kernel3x3 + self._pad_1x3_to_3x3_tensor(kernel1x3) + self._pad_3x1_to_3x3_tensor(kernel3x1) + kernelid
        biasdw = biasid + bias3x1 + bias1x3 + bias3x3

        return kerneldw, biasdw, pw_kernel_merged, pw_bias_merged




    def _pad_1x3_to_3x3_tensor(self, kernel1x3):
        if kernel1x3 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x3, [0, 0, 0, 2])

    def _pad_3x1_to_3x3_tensor(self, kernel3x1):
        if kernel3x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel3x1, [0, 2, 0, 0])

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam_dw'):
            return

        dw_kernel, dw_bias, pw_kernel, pw_bias = self.get_equivalent_kernel_bias()

        self.rbr_reparam_dw = nn.Conv2d(in_channels=self.rbr_3x3_dw.conv.in_channels,
                                        out_channels=self.rbr_3x3_dw.conv.out_channels,
                                        kernel_size=self.rbr_3x3_dw.conv.kernel_size, stride=self.rbr_3x3_dw.conv.stride,
                                        padding=self.rbr_3x3_dw.conv.padding, dilation=self.rbr_3x3_dw.conv.dilation,
                                        groups=self.rbr_3x3_dw.conv.groups, bias=True)

        self.rbr_reparam_pw = nn.Conv2d(in_channels=self.rbr_pw.conv.in_channels,
                                        out_channels=self.rbr_pw.conv.out_channels,
                                        kernel_size=3, stride=self.rbr_pw.conv.stride,
                                        padding=1, dilation=self.rbr_pw.conv.dilation,
                                        groups=self.rbr_pw.conv.groups, bias=True)

        self.rbr_reparam_dw.weight.data = dw_kernel
        self.rbr_reparam_dw.bias.data = dw_bias

        self.rbr_reparam_pw.weight.data = pw_kernel
        self.rbr_reparam_pw.bias.data = pw_bias

        self.__delattr__('rbr_3x3_dw')
        self.__delattr__('rbr_1x3_dw')
        self.__delattr__('rbr_3x1_dw')
        self.__delattr__('rbr_pw')
        self.__delattr__('rbr_avg')

        if hasattr(self, 'rbr_id'):
            self.__delattr__('rbr_id')

        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')

        self.deploy = True

class RepCFT(nn.Module):
    def __init__(self, cfgs, num_classes=1000, dropout=0.2, deploy=False):
        super(RepCFT, self).__init__()
        self.num_classes = num_classes
        self.cfgs = cfgs
        self.dropout = nn.Dropout(dropout)

        layers = []
        block = RepLGBlock

        for in_channels, out_channels, stride in cfgs:
            layers.append(RepLGBlock(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=3, stride=stride, deploy=deploy))

        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, self.num_classes)


    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

cfgs = [
    [3, 32, 2],
    [32, 64, 1],
    [64, 128, 2],
    [128, 128, 1],
    [128, 256, 2],
    [256, 256, 1],
    [256, 512, 2],
    [512, 512, 1],
    [512, 512, 1],
    [512, 512, 1],
    [512, 512, 1],
    [512, 512, 1],
    [512, 1024, 2],
    [1024, 1024, 1]]


def build(num_classes=1000, **kwargs):
    return RepCFT(cfgs=cfgs, num_classes=num_classes, **kwargs)

def RepCFT_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model

if __name__ == "__main__":
    net = build()
    net.eval()
    x = torch.randn(1, 3, 224, 224)
    net2 = RepCFT_model_convert(net)
    print(net2)
    x = net2(x)
    print(x.shape)


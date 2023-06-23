import torch
import torchvision.transforms as transforms

from copy import deepcopy
from torch import nn, Tensor
from torchvision import models
from torchvision.models import InceptionOutputs

softmax = nn.Softmax(dim=1)

normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
unnormalize = lambda x: 0.5 * x + 0.5


def get_prediction(model, input, normalize_im=False):
    """ Given model and input, returns predicted class and probabilites. """
    if normalize_im:
        input = normalize(input)
    output = model(input)
    _, pred = torch.max(output, dim=1)
    return pred, softmax(output)[:,pred]


def get_net(net_name, device):
    """ Loads pre-trained network. """
    if net_name == 'alexnet':
        net = models.alexnet(pretrained=True)
    elif net_name == 'vgg':
        net = models.vgg16(pretrained=True)
    elif net_name == 'resnet':
        net = models.resnet50(pretrained=True)
    elif net_name == 'densenet':
        net = models.densenet161(pretrained=True)
    elif net_name == 'inceptionv3':
        net = InceptionV3WithCorrectReLUs(models.inception_v3(pretrained=True))

    net.to(device)
    net.eval()
    return net

######################################################################################################################


class BasicConv2d(nn.Module):
    def __init__(self, basicConv2d):
        super().__init__()
        self.conv = basicConv2d._modules['conv']
        self.bn = basicConv2d._modules['bn']
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionV3WithCorrectReLUs(nn.Module):
    """ Wrapper class for Inceptionv3 model. """
    def __init__(self, other_model, verbose=False):
        super().__init__()
        self.model = deepcopy(other_model)
        self.replace_module(other_model, verbose)

    def replace_module(self, module, verbose=False):
        for mod_name, mod in module._modules.items():
            if verbose:
                print('processing {}'.format(mod_name))
            if isinstance(mod, models.inception.BasicConv2d):
                if verbose:
                    print('modifying {}'.format(mod_name))
                self.model._modules[mod_name] = BasicConv2d(mod)
            elif mod_name.startswith('Mixed') or mod_name.startswith('Aux'):
                for branch_mod_name, branch_mod in module._modules[mod_name]._modules.items():
                    if isinstance(branch_mod, models.inception.BasicConv2d):
                        if verbose:
                            print('modifying {}/{}'.format(mod_name, branch_mod_name))
                        self.model._modules[mod_name]._modules[branch_mod_name] = BasicConv2d(branch_mod)

    def forward(self, x: Tensor) -> InceptionOutputs:
        return self.model.forward(x)

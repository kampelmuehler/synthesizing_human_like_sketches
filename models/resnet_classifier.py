import torchvision
import torch.nn as nn
import torch.nn.functional as F


class ResnetClassifier(nn.Module):
    def __init__(self, num_classes=125):
        super(ResnetClassifier, self).__init__()
        resnet_model = torchvision.models.resnet34(pretrained=True)
        resnet_layers = list(resnet_model.children())
        Conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        Conv1.weight.data.copy_(resnet_layers[0].weight.data.mean(dim=1, keepdim=True))
        resnet_layers[0] = Conv1
        # take pretrained from Resnet, replace last layer
        resnet_layers[-1] = nn.Flatten(1)
        resnet_layers += [nn.Linear(512 * 1, num_classes)]
        self.resnet = nn.Sequential(*resnet_layers)

    def forward(self, x):
        return F.log_softmax(self.resnet(x), dim=1)


if __name__ == '__main__':
    ResnetClassifier()

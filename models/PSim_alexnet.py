import torchvision
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple


class PSim_Alexnet(nn.Module):

    def __init__(self, num_classes=125, train=True, with_classifier=False):
        super(PSim_Alexnet, self).__init__()
        self.train_mode = train
        self.with_classifier = with_classifier
        alexnet_model = torchvision.models.alexnet(pretrained=True)
        feature_layers = list(alexnet_model.features.children())
        self.Conv1 = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        # load weight from first channel of pretrained RGB model - narrow(dim, start, length)
        self.Conv1.weight.data.copy_(feature_layers[0].weight.data.narrow(1, 0, 1))
        self.Conv1.bias.data.copy_(feature_layers[0].bias.data)
        self.Conv2 = feature_layers[3]
        self.Conv3 = feature_layers[6]
        self.Conv4 = feature_layers[8]
        self.Conv5 = feature_layers[10]

        # take pretrained from Alexnet, replace last layer
        if train is True:
            linear = nn.Linear(4096, num_classes)
            # init with first num_classes weights from pretrained model
            linear.weight.data.copy_(alexnet_model.classifier.state_dict()['6.weight'].narrow(0, 0, num_classes))
            linear.bias.data.copy_(alexnet_model.classifier.state_dict()['6.bias'].narrow(0, 0, num_classes))
            alexnet_classifier = list(alexnet_model.classifier.children())
            self.classifier = nn.Sequential(nn.Dropout(p=0.5),
                                            *alexnet_classifier[1:3],
                                            nn.Dropout(p=0.5),
                                            *alexnet_classifier[4:-1],
                                            linear)

    def forward(self, x):
        conv1_activation = F.relu(self.Conv1(x))
        x = F.max_pool2d(conv1_activation, kernel_size=3, stride=2)
        conv2_activation = F.relu(self.Conv2(x))
        x = F.max_pool2d(conv2_activation, kernel_size=3, stride=2)
        conv3_activation = F.relu(self.Conv3(x))
        conv4_activation = F.relu(self.Conv4(conv3_activation))
        conv5_activation = F.relu(self.Conv5(conv4_activation))
        if self.with_classifier is True:
            x = F.max_pool2d(conv5_activation, kernel_size=3, stride=2)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            net_outputs = namedtuple("AlexnetActivations", ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
            return net_outputs(conv1_activation,
                               conv2_activation,
                               conv3_activation,
                               conv4_activation,
                               conv5_activation), F.log_softmax(x, dim=1)
        elif self.train_mode is True:
            x = F.max_pool2d(conv5_activation, kernel_size=3, stride=2)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return F.log_softmax(x, dim=1)
        else:
            net_outputs = namedtuple("AlexnetActivations", ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
            return net_outputs(conv1_activation, conv2_activation, conv3_activation, conv4_activation, conv5_activation)

    def load_weights(self, state_dict):
        # load only weights that are in the model (eg. if train=False, the classifier weights don't need to be loaded)

        model_dict = self.state_dict()
        # 1. filter out unnecessary keys
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(state_dict)
        # 3. load the new state dict
        self.load_state_dict(state_dict)

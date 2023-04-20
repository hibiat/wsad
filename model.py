import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


class Model_single(torch.nn.Module):
    def __init__(self, n_feature):
        super(Model_single, self).__init__()
        self.fc = nn.Linear(n_feature, n_feature)
        self.classifier = nn.Linear(n_feature, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.7)
        self.apply(weights_init)

    def forward(self, inputs, is_training=True):
        x = F.relu(self.fc(inputs))
        if is_training:
            x = self.dropout(x)
        return x, self.sigmoid(self.classifier(x))



class Model_mean(torch.nn.Module):
    def __init__(self, n_feature):
        super(Model_mean, self).__init__()
        self.fc = nn.Linear(n_feature, n_feature)
        self.conv1 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv3 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=5, stride=1,
                 padding=2, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv_b1 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv_b2 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')

        self.classifier = nn.Linear(n_feature, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.7)
        self.apply(weights_init)
        self.mean_pooling = nn.AvgPool2d((3, 1))

    def forward(self, inputs, is_training=True):
        inputs = inputs.permute(0, 2, 1)
        x_1 = F.relu(self.conv1(inputs)).permute(0, 2, 1).unsqueeze(2)
        x_2 = F.relu(self.conv2(inputs)).permute(0, 2, 1).unsqueeze(2)
        x_3 = F.relu(self.conv3(inputs)).permute(0, 2, 1).unsqueeze(2)
        x = torch.cat((x_1, x_2, x_3), dim=2)
        x = self.mean_pooling(x)
        x = x.squeeze(2)
        if is_training:
            x = self.dropout(x)
        return x, self.sigmoid(self.classifier(x))

class Model_sequence(torch.nn.Module):
    def __init__(self, n_feature):
        super(Model_sequence, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv3 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=5, stride=1,
                 padding=2, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv_b1 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')
        self.conv_b2 = nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros')

        self.classifier = nn.Linear(n_feature, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.7)
        self.apply(weights_init)

    def forward(self, inputs, is_training=True):
        inputs = inputs.permute(0, 2, 1)
        x_1 = F.relu(self.conv1(inputs))
        x_2 = F.relu(self.conv2(inputs))
        x_3 = F.relu(self.conv3(inputs))
        x = x_3 + x_2
        x = F.relu(self.conv_b2(x))
        x = x_1 + x
        x = F.relu(self.conv_b1(x))

        if is_training:
            x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x, self.sigmoid(self.classifier(x))


def model_generater(model_name, feature_size):
    if model_name == 'model_single':
        model = Model_single(feature_size) 
    elif model_name == 'model_mean':
        model = Model_mean(feature_size)
    elif model_name == 'model_sequence':
        model = Model_sequence(feature_size)
    else:
        raise ('model_name is out of option')
    return model
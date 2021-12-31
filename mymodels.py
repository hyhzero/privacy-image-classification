import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
import pdb
import numpy as np


class resnetBranch0Map(nn.Module):
    def __init__(self):
        super(resnetBranch0Map, self).__init__()
        model = models.resnet18(pretrained=True)
        for layer in model.parameters():
            layer.requires_grad = False

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class resnetBranch1Map(nn.Module):
    def __init__(self):
        super(resnetBranch1Map, self).__init__()
        model = models.resnet18(pretrained=True)
        for layer in model.parameters():
            layer.requires_grad = False

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        return x


class resnetBranch2Map(nn.Module):
    def __init__(self):
        super(resnetBranch2Map, self).__init__()
        model = models.resnet18(pretrained=True)
        for layer in model.parameters():
            layer.requires_grad = False

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class resnetBranch3Map(nn.Module):
    def __init__(self):
        super(resnetBranch3Map, self).__init__()
        model = models.resnet18(pretrained=True)
        for layer in model.parameters():
            layer.requires_grad = False

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool


        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x



class resnetBranch4Map(nn.Module):
    def __init__(self):
        super(resnetBranch4Map, self).__init__()
        model = models.resnet18(pretrained=True)
        for layer in model.parameters():
            layer.requires_grad = False

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class resnetBranch5Map(nn.Module):
    def __init__(self):
        super(resnetBranch5Map, self).__init__()
        model = models.resnet18(pretrained=True)
        for layer in model.parameters():
            layer.requires_grad = False

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x

# self-Attention
# class ScaledDotProductAttention(nn.Module):
#     ''' Scaled Dot-Product Attention '''
#     def __init__(self, temperature, attn_dropout=0.1):
#         super().__init__()
#         self.temperature = temperature
#         self.dropout = nn.Dropout(attn_dropout)
#         self.softmax = nn.Softmax(dim=2)

#     def forward(self, q, k, v, mask=None):

#         attn = torch.bmm(q, k.transpose(1, 2))
#         attn = attn / self.temperature

#         if mask is not None:
#             attn = attn.masked_fill(mask, -np.inf)

#         attn = self.softmax(attn)
#         attn = self.dropout(attn)
#         output = torch.bmm(attn, v)

#         return output, attn

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn



# Attention的输出拼接作为特征进行分类
class privacy_MSML(nn.Module):

    def __init__(self, ):
        super(privacy_MSML, self).__init__()
        self.lstms = nn.ModuleList([nn.LSTM(input_size=512, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
for _ in range(5)])

        d_k = 512
        self.attentions = nn.ModuleList([ScaledDotProductAttention(temperature=np.power(d_k, 0.5)) for _ in range(5)])
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.fc = nn.Linear(in_features=7680, out_features=2)

    def forward(self, x):
        outputs = []
        for i in range(5):
            inputs = x[:, i::5, :]
            inputs = inputs.contiguous()
            output, (h, c) = self.lstms[i](inputs)
            # attention
            output, atten = self.attentions[i](output, output, output)

            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        output, (h, c) = self.lstm(outputs)
        # attention
        output, atten = self.attention(output, output, output)
        output = output.view(output.size(0), -1)

        output = self.fc(output)

        return output


# Attention的输出拼接作为特征进行分类
class privacy_MLMS(nn.Module):

    def __init__(self):
        super(privacy_MLMS, self).__init__()
        self.lstms = nn.ModuleList([nn.LSTM(input_size=512, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)
for _ in range(3)])

        d_k = 512
        self.attentions = nn.ModuleList([ScaledDotProductAttention(temperature=np.power(d_k, 0.5)) for _ in range(3)])

        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.fc = nn.Linear(in_features=3840, out_features=2)

    def forward(self, x):
        outputs = []
        for i in range(3):
            inputs = x[:, 5*i:5*i+5, :]
            inputs = inputs.contiguous()
            output, (h, c) = self.lstms[i](inputs)
            # attention
            output, atten = self.attentions[i](output, output, output)

            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        output, (h, c) = self.lstm(outputs)
        # attention
        output, atten = self.attention(output, output, output)
        output = output.view(output.size(0), -1)

        output = self.fc(output)

        return output

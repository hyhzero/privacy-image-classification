import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb
import numpy as np


# branch0
class resnetBranch0(nn.Module):
    def __init__(self, model):
        super(resnetBranch0, self).__init__()
        # freeze resnet18
        for layer in model.parameters():
            layer.requires_grad = False

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        # branch0
        # self.branch0 = nn.Linear(64*56*56, 512)


        # branch1
        self.branch0 = nn.Sequential(
                nn.Linear(64 * 56 * 56, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )

        self.classifier0 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # brh0 = x.view(x.size(0), -1)
        # brh0 = self.branch0(brh0)

        # branch1
        brh0 = x.view(x.size(0), -1)
        brh0 = self.branch0(brh0)
        bcls0 = self.classifier0(brh0)

        return bcls0



# branch1
class resnetBranch1(nn.Module):
    def __init__(self, model):
        super(resnetBranch1, self).__init__()
        # freeze resnet18
        for layer in model.parameters():
            layer.requires_grad = False

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        # branch0
        # self.branch0 = nn.Linear(64*56*56, 512)

        self.layer1 = model.layer1

        # branch1
        self.branch1 = nn.Sequential(
                nn.Linear(64 * 56 * 56, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )

        self.classifier1 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # brh0 = x.view(x.size(0), -1)
        # brh0 = self.branch0(brh0)

        # branch1
        x = self.layer1(x)
        brh1 = x.view(x.size(0), -1)
        brh1 = self.branch1(brh1)
        bcls1 = self.classifier1(brh1)

        return bcls1


# branch2
class resnetBranch2(nn.Module):
    def __init__(self, model):
        super(resnetBranch2, self).__init__()
        # freeze branch1
        for layer in model.parameters():
            layer.requires_grad = False

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        # branch0
        # self.branch0 = nn.Linear(64*56*56, 512)

        self.layer1 = model.layer1
        self.layer2 = model.layer2

        # branch2
        self.branch2 = nn.Sequential(
                nn.Linear(128 * 28 * 28, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )

        self.classifier2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # brh0 = x.view(x.size(0), -1)
        # brh0 = self.branch0(brh0)

        # branch2
        x = self.layer1(x)
        x = self.layer2(x)

        brh2 = x.view(x.size(0), -1)
        brh2 = self.branch2(brh2)
        bcls2 = self.classifier2(brh2)

        return bcls2



# branch3
class resnetBranch3(nn.Module):
    def __init__(self, model):
        super(resnetBranch3, self).__init__()
        # freeze branch1， branch2
        for layer in model.parameters():
            layer.requires_grad = False

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        # branch0
        # self.branch0 = nn.Linear(64*56*56, 512)

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3


        # branch3
        self.branch3 = nn.Sequential(
                nn.Linear(256 * 14 * 14, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
               )

        self.classifier3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # brh0 = x.view(x.size(0), -1)
        # brh0 = self.branch0(brh0)

        # branch2
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        brh3 = x.view(x.size(0), -1)
        brh3 = self.branch3(brh3)
        bcls3 = self.classifier3(brh3)

        return bcls3



# branch4
class resnetBranch4(nn.Module):
    def __init__(self, model):
        super(resnetBranch4, self).__init__()
        # freeze branch1， branch2
        for layer in model.parameters():
            layer.requires_grad = False

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        # branch0
        # self.branch0 = nn.Linear(64*56*56, 512)

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        # branch4
        self.branch4 = nn.Sequential(
                nn.Linear(512 * 7 * 7, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )

        self.classifier4 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # brh0 = x.view(x.size(0), -1)
        # brh0 = self.branch0(brh0)

        # branch2
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        brh4 = x.view(x.size(0), -1)
        brh4 = self.branch4(brh4)
        bcls4 = self.classifier4(brh4)

        return bcls4

# branch5
class resnetBranch5(nn.Module):
    def __init__(self, model):
        super(resnetBranch5, self).__init__()
        # freeze branch1， branch2
        for layer in model.parameters():
            layer.requires_grad = False

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        # branch0
        # self.branch0 = nn.Linear(64*56*56, 512)

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
        # brh0 = x.view(x.size(0), -1)
        # brh0 = self.branch0(brh0)

        # branch2
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool

        return x

# fc
class resnet(nn.Module):
    def __init__(self, model):
        super(resnet, self).__init__()
        # freeze branch1， branch2
        for layer in model.parameters():
            layer.requires_grad = False

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        # branch0
        # self.branch0 = nn.Linear(64*56*56, 512)

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.avgpool = model.avgpool
        self.classifier = nn.Linear(512, 2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # brh0 = x.view(x.size(0), -1)
        # brh0 = self.branch0(brh0)

        # branch2
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        x = self.classifier(x.view(x.size(0), -1))

        return x



# self-Attention
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class mldrnet(nn.Module):
    def __init__(self, model):
        super(mldrnet, self).__init__()
        # freeze resnet18
        for layer in model.parameters():
            layer.requires_grad = False

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        # branch0
        self.branch0 = nn.Sequential(
                nn.Linear(64 * 56 * 56, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )

        # branch1
        self.layer1 = model.layer1
        self.branch1 = nn.Sequential(
            nn.Linear(64 * 56 * 56, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
        )


        # branch2
        self.layer2 = model.layer2
        self.branch2 = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
        )


        # branch3
        self.layer3 = model.layer3
        self.branch3 = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
        )


        # branch4
        self.layer4 = model.layer4
        self.branch4 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=2560, out_features=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # branch0
        brh0 = x.view(x.size(0), -1)
        brh0 = self.branch0(brh0)


        # branch1
        x = self.layer1(x)
        brh1 = x.view(x.size(0), -1)
        brh1 = self.branch1(brh1)

        # branch2
        x = self.layer2(x)
        brh2 = x.view(x.size(0), -1)
        brh2 = self.branch2(brh2)

        # branch3
        x = self.layer3(x)
        brh3 = x.view(x.size(0), -1)
        brh3 = self.branch3(brh3)

        # branch4
        x = self.layer4(x)
        brh4 = x.view(x.size(0), -1)
        brh4 = self.branch4(brh4)

        brhs = [brh0, brh1, brh2, brh3, brh4]
        brhs = [brh.unsqueeze(1) for brh in brhs]

        x = torch.cat(brhs, dim=1)
        x = x.contiguous()

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class cnnrnn(nn.Module):
    def __init__(self, model):
        super(cnnrnn, self).__init__()
        # freeze resnet18
        for layer in model.parameters():
            layer.requires_grad = False

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        # branch0
        self.branch0 = nn.Sequential(
                nn.Linear(64 * 56 * 56, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
               )

        # branch1
        self.layer1 = model.layer1
        self.branch1 = nn.Sequential(
            nn.Linear(64 * 56 * 56, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
        )


        # branch2
        self.layer2 = model.layer2
        self.branch2 = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
        )


        # branch3
        self.layer3 = model.layer3
        self.branch3 = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
        )


        # branch4
        self.layer4 = model.layer4
        self.branch4 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
        )

        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(in_features=1024, out_features=2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # branch0
        brh0 = x.view(x.size(0), -1)
        brh0 = self.branch0(brh0)


        # branch1
        x = self.layer1(x)
        brh1 = x.view(x.size(0), -1)
        brh1 = self.branch1(brh1)

        # branch2
        x = self.layer2(x)
        brh2 = x.view(x.size(0), -1)
        brh2 = self.branch2(brh2)

        # branch3
        x = self.layer3(x)
        brh3 = x.view(x.size(0), -1)
        brh3 = self.branch3(brh3)

        # branch4
        x = self.layer4(x)
        brh4 = x.view(x.size(0), -1)
        brh4 = self.branch4(brh4)

        brhs = [brh0, brh1, brh2, brh3, brh4]
        brhs = [brh.unsqueeze(1) for brh in brhs]

        x = torch.cat(brhs, dim=1)
        x = x.contiguous()

        output, (h, c) = self.lstm(x)
        output = torch.cat([output[:, 0, :], output[:, -1, :]], dim=1)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


class cnn_concat(nn.Module):

    def __init__(self):
        super(cnn_concat, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=2560, out_features=2)
        )


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class cnn_concat_scale(nn.Module):

    def __init__(self):
        super(cnn_concat_scale, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=2)
        )


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class cnn_branch(nn.Module):

    def __init__(self, index):
        super(cnn_branch, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=2)
        )
        self.index = index

    def forward(self, x):
        #x = x.view(x.size(0), -1)
        x = x[:, self.index, :]
        x = self.fc(x)

        return x



class bi_lstm(nn.Module):

    def __init__(self):
        super(bi_lstm, self).__init__()
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(in_features=1024, out_features=2)

    def forward(self, x):

        output, (h, c) = self.lstm(x)
        output = torch.cat([output[:, 0, :], output[:, -1, :]], dim=1)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


class privacy_MSML(nn.Module):

    def __init__(self):
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




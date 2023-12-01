import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleCNN(nn.Module):
    def __init__(self, class_num, model_config):
        super(SimpleCNN, self).__init__()
        self.conv1_channels, self.conv2_channels, self.dim, self.dim2 = model_config
        self.conv_sequence1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.conv1_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_sequence2 = nn.Sequential(
            nn.Conv2d(in_channels=self.conv1_channels, out_channels=self.conv2_channels, kernel_size=3, stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Linear(self.conv2_channels * 7 * 7, self.dim)
        self.fc2 = nn.Linear(self.dim, self.dim2)
        self.fc3 = nn.Linear(self.dim2, class_num)
        self.activation = nn.ReLU(inplace=True)
        self.feature_map = []

    def forward(self, x):
        x = self.conv_sequence1(x)
        x = self.conv_sequence2(x)
        x = x.view(-1, self.conv2_channels * 7 * 7)
        x = self.activation(self.fc1(x))
        self.feature_map = [x.clone()]
        x = self.activation(self.fc2(x))
        return self.fc3(x)


class DotProductCNN(nn.Module):
    def __init__(self, class_num, model_config):
        super(DotProductCNN, self).__init__()
        self.conv1_channels, self.conv2_channels, self.dim, self.dim2 = model_config
        self.conv_sequence1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.conv1_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_sequence2 = nn.Sequential(
            nn.Conv2d(in_channels=self.conv1_channels, out_channels=self.conv2_channels, kernel_size=3, stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        dim_half = int(self.dim / 2)
        self.fc1 = nn.Linear(self.conv2_channels * 7 * 7, dim_half)
        self.fc2 = nn.Linear(self.conv2_channels * 7 * 7, dim_half)
        self.fc3 = nn.Linear(dim_half, class_num)
        self.activation = nn.ReLU(inplace=True)
        self.feature_map = []

    def forward(self, x):
        x = self.conv_sequence1(x)
        x = self.conv_sequence2(x)
        x = x.view(-1, self.conv2_channels * 7 * 7)
        x1 = self.activation(self.fc1(x))
        x2 = self.activation(self.fc2(x))
        x = x1 * x2
        self.feature_map = [x1.clone(), x2.clone(), x.clone()]
        return self.fc3(x)


class CrossProductCNN(nn.Module):
    def __init__(self, class_num, model_config):
        super(CrossProductCNN, self).__init__()
        self.conv1_channels, self.conv2_channels, self.dim, self.dim2 = model_config
        self.conv_sequence1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.conv1_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_sequence2 = nn.Sequential(
            nn.Conv2d(in_channels=self.conv1_channels, out_channels=self.conv2_channels, kernel_size=3, stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        dim_half = int(self.dim / 2)
        self.fc1 = nn.Linear(self.conv2_channels * 7 * 7, dim_half)
        self.fc2 = nn.Linear(self.conv2_channels * 7 * 7, dim_half)
        self.fc3 = nn.Linear(dim_half * dim_half, class_num)
        self.activation = nn.ReLU(inplace=True)
        self.feature_map = []

    def forward(self, x):
        x = self.conv_sequence1(x)
        x = self.conv_sequence2(x)
        x = x.view(-1, self.conv2_channels * 7 * 7)
        x1 = self.activation(self.fc1(x))
        x2 = self.activation(self.fc2(x))
        x = torch.einsum('ik,ij->ikj', x1, x2).view(-1, int(self.dim / 2) * int(self.dim / 2))
        self.feature_map = [x1.clone(), x2.clone(), x.clone()]

        return self.fc3(x)


class ISTALayer(nn.Module):
    def __init__(self, in_features, out_features, eta, lambda1):
        super(ISTALayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eta = eta
        self.lambda1 = lambda1
        self.dictionary = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.dictionary, a=math.sqrt(5))

    def forward(self, input):
        Z = F.linear(input, self.dictionary)
        grad_1 = F.linear(Z, self.dictionary.t(), bias=None)
        grad_2 = F.linear(input, self.dictionary.t(), bias=None)
        grad_update = self.eta * (grad_2 - grad_1) - self.eta * self.lambda1
        output = F.relu(input + grad_update)
        return output


class DotProductSparseCNN(nn.Module):
    def __init__(self, class_num, model_config, lambda_sparse=0.01):
        super(DotProductSparseCNN, self).__init__()
        self.conv1_channels, self.conv2_channels, self.dim, self.dim2 = model_config
        self.conv_sequence1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.conv1_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_sequence2 = nn.Sequential(
            nn.Conv2d(in_channels=self.conv1_channels, out_channels=self.conv2_channels, kernel_size=3, stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        dim_half = int(self.dim / 2)
        self.fc1 = nn.Linear(self.conv2_channels * 7 * 7, dim_half)
        self.fc2 = nn.Linear(self.conv2_channels * 7 * 7, dim_half)
        self.fc3 = nn.Linear(dim_half, class_num)
        self.ista1 = ISTALayer(dim_half, dim_half, eta=0.1, lambda1=lambda_sparse)
        self.ista2 = ISTALayer(dim_half, dim_half, eta=0.1, lambda1=lambda_sparse)
        self.feature_map = []

    def forward(self, x):
        x = self.conv_sequence1(x)
        x = self.conv_sequence2(x)
        x = x.view(-1, self.conv2_channels * 7 * 7)
        x1 = self.ista1(self.fc1(x))
        x2 = self.ista2(self.fc2(x))
        x = x1 * x2
        self.feature_map = [x1.clone(), x2.clone(), x.clone()]
        return self.fc3(x)


class CrossProductSparseCNN(nn.Module):
    def __init__(self, class_num, model_config, lambda_sparse=0.01):
        super(CrossProductSparseCNN, self).__init__()
        self.conv1_channels, self.conv2_channels, self.dim, self.dim2 = model_config
        self.conv_sequence1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.conv1_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_sequence2 = nn.Sequential(
            nn.Conv2d(in_channels=self.conv1_channels, out_channels=self.conv2_channels, kernel_size=3, stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        dim_half = int(self.dim / 2)
        self.fc1 = nn.Linear(self.conv2_channels * 7 * 7, dim_half)
        self.fc2 = nn.Linear(self.conv2_channels * 7 * 7, dim_half)
        self.fc3 = nn.Linear(dim_half * dim_half, class_num)
        self.ista1 = ISTALayer(dim_half, dim_half, eta=0.1, lambda1=lambda_sparse)
        self.ista2 = ISTALayer(dim_half, dim_half, eta=0.1, lambda1=lambda_sparse)
        self.feature_map = []

    def forward(self, x):
        x = self.conv_sequence1(x)
        x = self.conv_sequence2(x)
        x = x.view(-1, self.conv2_channels * 7 * 7)
        x1 = self.ista1(self.fc1(x))
        x2 = self.ista2(self.fc2(x))
        x = torch.einsum('ik,ij->ikj', x1, x2).view(-1, int(self.dim / 2) * int(self.dim / 2))
        self.feature_map = [x1.clone(), x2.clone(), x.clone()]
        return self.fc3(x)


class CrossProductAsymmetricCNN(nn.Module):
    def __init__(self, class_num, model_config):
        super(CrossProductAsymmetricCNN, self).__init__()
        self.conv1_channels, self.conv2_channels, self.dim, self.dim2 = model_config
        self.conv_sequence1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.conv1_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_sequence2 = nn.Sequential(
            nn.Conv2d(in_channels=self.conv1_channels, out_channels=self.conv2_channels, kernel_size=3, stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        dim_one_fourth = int(self.dim / 4)
        self.fc1 = nn.Linear(self.conv2_channels * 7 * 7, self.dim - dim_one_fourth)
        self.fc2 = nn.Linear(self.conv2_channels * 7 * 7, dim_one_fourth)
        self.fc3 = nn.Linear((self.dim - dim_one_fourth) * dim_one_fourth, class_num)
        self.activation = nn.ReLU(inplace=True)
        self.feature_map = []

    def forward(self, x):
        x = self.conv_sequence1(x)
        x = self.conv_sequence2(x)
        x = x.view(-1, self.conv2_channels * 7 * 7)
        x1 = self.activation(self.fc1(x))
        x2 = self.activation(self.fc2(x))
        x = torch.einsum('ik,ij->ikj', x1, x2).view(-1, int(self.dim - int(self.dim / 4)) * int(self.dim / 4))
        self.feature_map = [x1.clone(), x2.clone(), x.clone()]

        return self.fc3(x)

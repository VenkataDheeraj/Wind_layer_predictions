import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(ResidualBlock, self).__init__()
        
        self.linear1 = nn.Linear(input_dims, output_dims, bias = False)
        nn.init.kaiming_normal_(self.linear1.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        self.ln1 = nn.LayerNorm(output_dims, elementwise_affine=True)
        self.leaky_relu = nn.LeakyReLU(0.01)
        
        self.linear2 = nn.Linear(output_dims, output_dims, bias = False)
        nn.init.kaiming_normal_(self.linear2.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        self.ln2 = nn.LayerNorm(output_dims, elementwise_affine=True)

        if input_dims != output_dims:
            self.residual = nn.Linear(input_dims, output_dims)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        
        out = self.linear1(x)
        out = self.ln1(out)
        out = self.leaky_relu(out)
        
        out = self.linear2(out)
        out = self.ln2(out)
        
        out += residual
        out = self.leaky_relu(out)
        return out

class ResNetRegressor(nn.Module):
    def __init__(self, input_dim=100, output_dim=8, hidden_dims=[100,100,50,50]):
        super(ResNetRegressor, self).__init__()
        input_layer = nn.Linear(input_dim, hidden_dims[0], bias = False)
        nn.init.kaiming_normal_(input_layer.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        ln_input = nn.LayerNorm(hidden_dims[0], elementwise_affine = True)
        leaky_relu = nn.LeakyReLU()
        
        self.model_layers = [input_layer, ln_input, leaky_relu]
        
        for i in range(len(hidden_dims) - 1):
            res_block = ResidualBlock(hidden_dims[i], hidden_dims[i + 1])
            self.model_layers.append(res_block)
            
        output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.model_layers.append(output_layer)
        
        self.architecture = nn.Sequential(*self.model_layers)

    def forward(self, x):
        x = self.architecture(x)
        x = F.sigmoid(x)
        return x

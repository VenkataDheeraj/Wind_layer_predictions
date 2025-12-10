# +
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet_models import ResNetRegressor


# -

class CNN_Hybrid_Domain_64(nn.Module):
    def __init__(self, fc_2_nodes=8000, latent_dims=1000, regressor_hidden_dims = [500,500,200,100], output_dims=8, device= None):
        super(CNN_Hybrid_Domain_64, self).__init__()
        self.device = device
        self.fc_1_nodes_pixel = 6400
        self.fc_1_nodes_freq = 100 * 2 * 2 * 4  # = 1600
        self.fc_2_nodes = fc_2_nodes
        self.latent_dims = latent_dims
        self.output_dims = output_dims
        self.regressor_hidden_dims = regressor_hidden_dims
        self.regressor = ResNetRegressor(input_dim=self.latent_dims, output_dim=self.output_dims, hidden_dims=regressor_hidden_dims)

        # CONV Pixel Encoder
        
        pconvLayer1 = nn.Conv3d(1, 16, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
        nn.init.kaiming_normal_(pconvLayer1.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        plreluLayer1 = nn.LeakyReLU(0.01)
        pmaxpoolLayer1 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))

        
        pconvLayer2 = nn.Conv3d(16, 64, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
        nn.init.kaiming_normal_(pconvLayer2.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        plreluLayer2 = nn.LeakyReLU(0.01)
        pmaxpoolLayer2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))

        
        pconvLayer3 = nn.Conv3d(64, 100, kernel_size=(3,3,3), stride=(1,2,2), padding=1)
        nn.init.kaiming_normal_(pconvLayer3.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        plreluLayer3 = nn.LeakyReLU(0.01)        

        # Dense Pixel Encoder
        pflatten = nn.Flatten()
        pdense4 = nn.Linear(self.fc_1_nodes_pixel,self.fc_2_nodes)
        nn.init.kaiming_normal_(pdense4.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        plreluLayer4 = nn.LeakyReLU(0.01)
        pdense5 = nn.Linear(self.fc_2_nodes,self.latent_dims)
        nn.init.kaiming_normal_(pdense5.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        plreluLayer5 = nn.LeakyReLU(0.01)

        self.conv_encoder_pixel = nn.Sequential(
                pconvLayer1, plreluLayer1, pmaxpoolLayer1,
                pconvLayer2, plreluLayer2, pmaxpoolLayer2,
                pconvLayer3, plreluLayer3
                )
        self.dense_encoder_pixel = nn.Sequential(
                pflatten,
                pdense4, plreluLayer4,
                pdense5, plreluLayer5
                )
        
        # CONV Freq Encoder
        
        fconvLayer1 = nn.Conv3d(2, 16, kernel_size=3, stride=2, padding=1)
        nn.init.kaiming_normal_(fconvLayer1.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        flreluLayer1 = nn.LeakyReLU(0.01)
        fmaxpoolLayer1 = nn.MaxPool3d(kernel_size=2, stride=2)

        
        fconvLayer2 = nn.Conv3d(16, 64, kernel_size=3, stride=2, padding=1)
        nn.init.kaiming_normal_(fconvLayer2.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        flreluLayer2 = nn.LeakyReLU(0.01)
        fmaxpoolLayer2 = nn.MaxPool3d(kernel_size=2, stride=2)

        
        fconvLayer3 = nn.Conv3d(64, 100, kernel_size=3, stride=(1,1,1), padding=1)
        nn.init.kaiming_normal_(fconvLayer3.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        flreluLayer3 = nn.LeakyReLU(0.01)        
        
        # Dense Freq Encoder
        fflatten = nn.Flatten()
        fdense4 = nn.Linear(self.fc_1_nodes_freq,self.fc_2_nodes)
        nn.init.kaiming_normal_(fdense4.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        flreluLayer4 = nn.LeakyReLU(0.01)
        fdense5 = nn.Linear(self.fc_2_nodes,self.latent_dims)
        nn.init.kaiming_normal_(fdense5.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        flreluLayer5 = nn.LeakyReLU(0.01)

        self.conv_encoder_freq = nn.Sequential(
                fconvLayer1, flreluLayer1, fmaxpoolLayer1,
                fconvLayer2, flreluLayer2, fmaxpoolLayer2,
                fconvLayer3, flreluLayer3
                )
        self.dense_encoder_freq = nn.Sequential(
                fflatten,
                fdense4, flreluLayer4,
                fdense5, flreluLayer5
                )
        self.project = nn.Linear(2*latent_dims, latent_dims, bias = False)
        self.ln = nn.LayerNorm(latent_dims, elementwise_affine=True)
        
    def encode_pixel(self, x):
        x = self.conv_encoder_pixel(x)
        x = self.dense_encoder_pixel(x)
        return x
    
    def encode_freq(self, x):
        x = self.conv_encoder_freq(x)
        x = self.dense_encoder_freq(x)
        return x

    def load_weights(self, old_model_state_dict, new_model_state_dict= None):
        own_state = self.state_dict()
        for name, param in old_model_state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                param = param.data
            own_state[name].copy_(param)
        return

    def forward(self,x_pixel, x_freq):
        z_pixel = self.encode_pixel(x_pixel)
        z_freq = self.encode_freq(x_freq)
        z_cat = torch.cat([z_pixel, z_freq], dim = 1)
        z = self.ln(self.project(z_cat))
        y_hat = self.regressor(z)
        
        return z, y_hat
        

class CNN_Hybrid_Domain_64_v2(nn.Module):
    def __init__(self, fc_2_nodes=8000, latent_dims=1000, regressor_hidden_dims = [500,500,200,100], output_dims=8, device= None):
        super(CNN_Hybrid_Domain_64_v2, self).__init__()
        self.device = device
        self.fc_1_nodes = 6400
        self.fc_2_nodes = fc_2_nodes
        self.latent_dims = latent_dims
        self.output_dims = output_dims
        self.regressor_hidden_dims = regressor_hidden_dims
        self.regressor = ResNetRegressor(input_dim=self.latent_dims, output_dim=self.output_dims, hidden_dims=regressor_hidden_dims)

        # CONV Encoder
        
        convLayer1 = nn.Conv3d(3, 16, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
        nn.init.kaiming_normal_(convLayer1.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        lreluLayer1 = nn.LeakyReLU(0.01)
        maxpoolLayer1 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))

        
        convLayer2 = nn.Conv3d(16, 64, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
        nn.init.kaiming_normal_(convLayer2.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        lreluLayer2 = nn.LeakyReLU(0.01)
        maxpoolLayer2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))

        
        convLayer3 = nn.Conv3d(64, 100, kernel_size=(3,3,3), stride=(1,2,2), padding=1)
        nn.init.kaiming_normal_(convLayer3.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        lreluLayer3 = nn.LeakyReLU(0.01)        

        #Dense Encoder
        flatten = nn.Flatten()
        dense4 = nn.Linear(self.fc_1_nodes,self.fc_2_nodes)
        nn.init.kaiming_normal_(dense4.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        lreluLayer4 = nn.LeakyReLU(0.01)
        dense5 = nn.Linear(self.fc_2_nodes,self.latent_dims)
        nn.init.kaiming_normal_(dense5.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        lreluLayer5 = nn.LeakyReLU(0.01)

        self.conv_encoder = nn.Sequential(
                convLayer1, lreluLayer1, maxpoolLayer1,
                convLayer2, lreluLayer2, maxpoolLayer2,
                convLayer3, lreluLayer3
                )
        self.dense_encoder = nn.Sequential(
                flatten,
                dense4, lreluLayer4,
                dense5, lreluLayer5
                )
        
    def encode(self, x):
        x = self.conv_encoder(x)
        x = self.dense_encoder(x)
        return x

    def load_weights(self, old_model_state_dict, new_model_state_dict= None):
        own_state = self.state_dict()
        for name, param in old_model_state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                param = param.data
            own_state[name].copy_(param)
        return
    def forward(self,x_pixel, x_freq):
        x = torch.cat((x_pixel, x_freq), dim=1)
        z = self.encode(x)
        y_hat = self.regressor(z)
        return z, y_hat


class CNN_Hybrid_Domain_64_v3(nn.Module):
    def __init__(self, fc_2_nodes=8000, latent_dims=1000, regressor_hidden_dims = [500,500,200,100], output_dims=8, device= None):
        super(CNN_Hybrid_Domain_64_v3, self).__init__()
        self.device = device
        self.fc_1_nodes_pixel = 6400
        self.fc_1_nodes_freq = 100 * 2 * 2 * 4  # = 1600
        self.fc_2_nodes = fc_2_nodes
        self.latent_dims = latent_dims
        self.output_dims = output_dims
        self.regressor_hidden_dims = regressor_hidden_dims
        self.regressor_pixel = ResNetRegressor(input_dim=self.latent_dims, output_dim=self.output_dims, hidden_dims=regressor_hidden_dims)
        self.regressor_freq = ResNetRegressor(input_dim=self.latent_dims, output_dim=self.output_dims, hidden_dims=regressor_hidden_dims)
        
        # CONV Pixel Encoder
        
        pconvLayer1 = nn.Conv3d(1, 16, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
        nn.init.kaiming_normal_(pconvLayer1.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        plreluLayer1 = nn.LeakyReLU(0.01)
        pmaxpoolLayer1 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))

        
        pconvLayer2 = nn.Conv3d(16, 64, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
        nn.init.kaiming_normal_(pconvLayer2.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        plreluLayer2 = nn.LeakyReLU(0.01)
        pmaxpoolLayer2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))

        
        pconvLayer3 = nn.Conv3d(64, 100, kernel_size=(3,3,3), stride=(1,2,2), padding=1)
        nn.init.kaiming_normal_(pconvLayer3.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        plreluLayer3 = nn.LeakyReLU(0.01)        

        # Dense Pixel Encoder
        pflatten = nn.Flatten()
        pdense4 = nn.Linear(self.fc_1_nodes_pixel,self.fc_2_nodes)
        nn.init.kaiming_normal_(pdense4.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        plreluLayer4 = nn.LeakyReLU(0.01)
        pdense5 = nn.Linear(self.fc_2_nodes,self.latent_dims)
        nn.init.kaiming_normal_(pdense5.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        plreluLayer5 = nn.LeakyReLU(0.01)

        self.conv_encoder_pixel = nn.Sequential(
                pconvLayer1, plreluLayer1, pmaxpoolLayer1,
                pconvLayer2, plreluLayer2, pmaxpoolLayer2,
                pconvLayer3, plreluLayer3
                )
        self.dense_encoder_pixel = nn.Sequential(
                pflatten,
                pdense4, plreluLayer4,
                pdense5, plreluLayer5
                )
        
        # CONV Freq Encoder
        
        fconvLayer1 = nn.Conv3d(2, 16, kernel_size=3, stride=2, padding=1)
        nn.init.kaiming_normal_(fconvLayer1.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        flreluLayer1 = nn.LeakyReLU(0.01)
        fmaxpoolLayer1 = nn.MaxPool3d(kernel_size=2, stride=2)

        
        fconvLayer2 = nn.Conv3d(16, 64, kernel_size=3, stride=2, padding=1)
        nn.init.kaiming_normal_(fconvLayer2.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        flreluLayer2 = nn.LeakyReLU(0.01)
        fmaxpoolLayer2 = nn.MaxPool3d(kernel_size=2, stride=2)

        
        fconvLayer3 = nn.Conv3d(64, 100, kernel_size=3, stride=(1,1,1), padding=1)
        nn.init.kaiming_normal_(fconvLayer3.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        flreluLayer3 = nn.LeakyReLU(0.01)        
        
        # Dense Freq Encoder
        fflatten = nn.Flatten()
        fdense4 = nn.Linear(self.fc_1_nodes_freq,self.fc_2_nodes)
        nn.init.kaiming_normal_(fdense4.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        flreluLayer4 = nn.LeakyReLU(0.01)
        fdense5 = nn.Linear(self.fc_2_nodes,self.latent_dims)
        nn.init.kaiming_normal_(fdense5.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        flreluLayer5 = nn.LeakyReLU(0.01)

        self.conv_encoder_freq = nn.Sequential(
                fconvLayer1, flreluLayer1, fmaxpoolLayer1,
                fconvLayer2, flreluLayer2, fmaxpoolLayer2,
                fconvLayer3, flreluLayer3
                )
        self.dense_encoder_freq = nn.Sequential(
                fflatten,
                fdense4, flreluLayer4,
                fdense5, flreluLayer5
                )
        self.project = nn.Linear(2*latent_dims, latent_dims, bias = False)
        self.ln = nn.LayerNorm(latent_dims, elementwise_affine=True)
        
    def encode_pixel(self, x):
        x = self.conv_encoder_pixel(x)
        x = self.dense_encoder_pixel(x)
        return x
    
    def encode_freq(self, x):
        x = self.conv_encoder_freq(x)
        x = self.dense_encoder_freq(x)
        return x

    def load_weights(self, old_model_state_dict, new_model_state_dict= None):
        own_state = self.state_dict()
        for name, param in old_model_state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                param = param.data
            own_state[name].copy_(param)
        return

    def forward(self,x_pixel, x_freq):
        z_pixel = self.encode_pixel(x_pixel)
        z_freq = self.encode_freq(x_freq)
        y_pixel = self.regressor_pixel(z_pixel)
        y_freq = self.regressor_freq(z_freq)
        y_hat = (y_pixel + y_freq)/2
        z = (z_pixel + z_freq)/2
        return z, y_hat
        
class CNN_Hybrid_Domain_64_v2_vicreg(nn.Module):
    def __init__(self, fc_2_nodes=8000, latent_dims=1000, regressor_hidden_dims = [500,500,200,100], output_dims=8, device= None):
        super(CNN_Hybrid_Domain_64_v2_vicreg, self).__init__()
        self.device = device
        self.fc_1_nodes = 6400
        self.fc_2_nodes = fc_2_nodes
        self.latent_dims = latent_dims
        self.output_dims = output_dims
        self.regressor_hidden_dims = regressor_hidden_dims
        self.regressor = ResNetRegressor(input_dim=self.latent_dims, output_dim=self.output_dims, hidden_dims=regressor_hidden_dims)

        # CONV Encoder
        
        convLayer1 = nn.Conv3d(3, 16, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
        nn.init.kaiming_normal_(convLayer1.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        lreluLayer1 = nn.LeakyReLU(0.01)
        maxpoolLayer1 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))

        
        convLayer2 = nn.Conv3d(16, 64, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
        nn.init.kaiming_normal_(convLayer2.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        lreluLayer2 = nn.LeakyReLU(0.01)
        maxpoolLayer2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))

        
        convLayer3 = nn.Conv3d(64, 100, kernel_size=(3,3,3), stride=(1,2,2), padding=1)
        nn.init.kaiming_normal_(convLayer3.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        lreluLayer3 = nn.LeakyReLU(0.01)        

        #Dense Encoder
        flatten = nn.Flatten()
        dense4 = nn.Linear(self.fc_1_nodes,self.fc_2_nodes)
        nn.init.kaiming_normal_(dense4.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        lreluLayer4 = nn.LeakyReLU(0.01)
        dense5 = nn.Linear(self.fc_2_nodes,self.latent_dims)
        nn.init.kaiming_normal_(dense5.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        lreluLayer5 = nn.LeakyReLU(0.01)
        
        #Dense Decoder
        dense6 = nn.Linear(self.latent_dims,1200)
        nn.init.kaiming_normal_(dense6.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
        lreluLayer6 = nn.LeakyReLU(0.01)

        self.conv_encoder = nn.Sequential(
                convLayer1, lreluLayer1, maxpoolLayer1,
                convLayer2, lreluLayer2, maxpoolLayer2,
                convLayer3, lreluLayer3
                )
        self.dense_encoder = nn.Sequential(
                flatten,
                dense4, lreluLayer4,
                dense5, lreluLayer5
                )
        
        self.dense_decoder = nn.Sequential(dense6, lreluLayer6)
        
    def encode(self, x):
        x = self.conv_encoder(x)
        x = self.dense_encoder(x)
        return x
        
    def decode(self, x):
        x = self.dense_decoder(x)
        return x
        
    def load_weights(self, old_model_state_dict, new_model_state_dict= None):
        own_state = self.state_dict()
        for name, param in old_model_state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                param = param.data
            own_state[name].copy_(param)
        return
    def forward(self,x_pixel, x_freq):
        x = torch.cat((x_pixel, x_freq), dim=1)
        z = self.encode(x)
        h = self.decode(z)
        y_hat = self.regressor(z)
        return h, y_hat

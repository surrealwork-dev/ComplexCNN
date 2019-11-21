import torch
import time
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import numpy as np
import GrassmannAverage as gm


class autoencoder(nn.Module):
    def __init__(self, num_frames, num_channels, latent_variable_size):
        super(autoencoder, self).__init__()

        self.num_frames = num_frames
        self.num_channels = num_channels
        self.latent_variable_size = latent_variable_size

        self.eConv = nn.Conv1d(self.num_channels, self.num_channels, 2, 1)
        self.bn = nn.BatchNorm1d(self.num_channels)

        #Grassmann average - PCA approximation in latent space.
        self.pca = gm.GrassmannAverage(self.num_frames, 20)
        self.fc1 = nn.Linear(20, self.latent_variable_size)

        # decoder
        self.d1 = nn.Linear(self.latent_variable_size, self.num_channels)
        self.dConv = nn.ConvTranspose1d(self.num_channels, self.num_channels, 2, 1)

        # activations
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        shape_outputs = []
        shape_outputs.append(x.shape)
        h1 = self.elu(self.bn(self.eConv(x)))
        shape_outputs.append(h1.shape)
        h2 = self.elu(self.bn(self.eConv(h1)))
        shape_outputs.append(h2.shape)
        h3 = self.elu(self.bn(self.eConv(h2)))
        shape_outputs.append(h3.shape)
        h4 = self.elu(self.bn(self.eConv(h3)))
        shape_outputs.append(h4.shape)
        h5 = h4.view(-1, self.num_channels)
        shape_outputs.append(h5.shape)

        y, self.weight_penalty = self.pca(h5)

        return self.fc1(y), shape_outputs

    def decode(self, z, shape_outputs, h5shape):
        h1 = self.relu(self.d1(z))
        h1 = nn.Linear(h1.shape[0], shape_outputs[-1][-1])(h1.t())
        h1 = h1.view(self.num_frames, self.num_channels, -1)
        h2 = self.elu(self.bn(self.dConv(h1)))
        h3 = self.elu(self.bn(self.dConv(h2.alpha)))
        h4 = self.elu(self.bn(self.dConv(h3.alpha)))
        h5 = self.elu(self.bn(self.dConv(h4.alpha)))
        assert h5.alpha.shape == shape_outputs[0]

        return self.sigmoid(h5.alpha)

    def forward(self, x):
        latent, shape_outputs = self.encode(x)
        res = self.decode(latent, shape_outputs)
        return res, self.weight_penalty

def make_model(num_frames, num_channels, latent_variable_size):
    model = autoencoder(num_frames=num_frames, num_channels=num_channels, latent_variable_size=latent_variable_size)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Model Parameters: ", params)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    return model, optimizer

#MSE loss
def loss_function(recon_x, x):
    mse_loss = torch.mean((recon_x-x)**2)
    return mse_loss

def train(X_train_np, latent_variable_size=50, num_epochs=100):
    X_train = torch.from_numpy(X_train_np)
    
    m, opt = make_model(num_frames=X_train_np.shape[0], num_channels=X_train_np.shape[1], latent_variable_size=latent_variable_size)

    loss_track = []
    recon_batch = None

    try:
        for i in range(num_epochs):
            start = time.time() 
            opt.zero_grad()
            recon_batch, weight_penalty = m(X_train)
            loss = loss_function(recon_batch, X_train)
            loss = loss+weight_penalty
            print('Iteration ', i, ': ', loss-weight_penalty)
            loss.backward()
            opt.step()
            end = time.time()
            print('Iteration ', i, ' time: ', end-start)

    except KeyboardInterrupt:
        pass
            
    return recon_batch, loss_track


#vid,loss = train()
#np.save('out_m2', vid.detach().numpy())
#np.save('loss_m2', np.array(loss))
#torch.save(model, 'model2')

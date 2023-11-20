import torch.nn as nn
import torchvision.models
from torchvision.models.resnet import ResNet18_Weights

class encoder_decoder:

    encoder = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    nfeature = encoder.fc.out_features

    frontend = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(nfeature, 2)
    )
class model(nn.Module):

    def __init__(self, encoder=None, type=True):
        super(model, self).__init__()
        self.encoder = encoder
        self.frontend = encoder_decoder.frontend

        if self.encoder is None:
            # if type:
            #     self.encoder = encoder_decoder.encoder
            # else:
            self.encoder = encoder_decoder.encoder

            self.init_encoder_weights(mean=0.0, std=0.01)
            self.init_frontend_weights(mean=0.0, std=0.01)

        self.mse_loss = nn.MSELoss()

    def init_encoder_weights(self, mean, std):
        for param in self.encoder.parameters():
            nn.init.normal_(param, mean=mean, std=std)

    def init_frontend_weights(self, mean, std):
        for param in self.frontend.parameters():
            nn.init.normal_(param, mean=mean, std=std)

    def forward(self, X):
        X = self.encoder(X)
        X = self.frontend(X)
        return X

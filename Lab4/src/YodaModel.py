import torch.nn as nn
import torchvision.models


class encoder_decoder:
    encoder = torchvision.models.resnet18()

    nfeature = encoder.fc.in_features
    encoder.fc = nn.Linear(nfeature,2)

class model(nn.Module):

    def __init__(self, encoder=None):
        super(model, self).__init__()
        self.encoder = encoder

        if self.encoder is None:
            self.encoder = encoder_decoder.encoder
            self.init_encoder_weights(mean=0.0, std=0.01)

        self.mse_loss = nn.MSELoss()

    def init_encoder_weights(self, mean, std):
        for param in self.encoder.parameters():
            nn.init.normal_(param, mean=mean, std=std)


    def forward(self, X):
        X = self.encoder(X)
        return X

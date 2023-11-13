import torch.nn as nn
import torchvision.models


class encoder_decoder:
    encoder = torchvision.models.resnet18()

    #Selected frontend, delete others
    frontend = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(512 * 4 * 4, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2),  # 100 b/c 100 images output at output layer
        # nn.ReLU(),
        nn.Softmax(dim=1)
    )

class model(nn.Module):

    def __init__(self, encoder, decoder=None):
        super(model, self).__init__()
        self.encoder = encoder
        # freeze encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.decoder = decoder
        #   if no decoder loaded, then initialize with random weights
        if self.decoder == None:
            # self.decoder = _decoder
            self.decoder = encoder_decoder.frontend
            self.init_decoder_weights(mean=0.0, std=0.01)

        self.mse_loss = nn.MSELoss()

    def init_decoder_weights(self, mean, std):
        for param in self.decoder.parameters():
            nn.init.normal_(param, mean=mean, std=std)

    def decode(self, X): #take in the flattened image of relu4_1 feature map
        return self.decoder(X)

    def forward(self, X):
        X = self.encoder(X)
        output = X.view(-1, 512*4*4)
        # print(relu_4.shape)
        return self.decode(output)


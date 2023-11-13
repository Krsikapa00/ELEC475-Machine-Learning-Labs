import torch.nn as nn


class encoder_decoder:
    encoder = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
    )

    #Selected frontend, delete others
    frontend = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(512 * 4 * 4, 2048),
        nn.ReLU(),
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 100),  # 100 b/c 100 images output at output layer
        # nn.ReLU(),
        nn.Softmax(dim=1)
    )
    # Selected Modded frontend, delete others
    frontend_10 = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(512 * 4 * 4, 2048),
        nn.ReLU(),
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),  # 100 b/c 100 images output at output layer
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


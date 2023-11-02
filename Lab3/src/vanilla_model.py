import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


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
    frontend = nn.Sequential(
        nn.Linear(512*4*4, 256),
        nn.ReLU(),
        nn.Linear(256,10), #100 b/c 100 images output at output layer
        #nn.ReLU(),
        nn.Softmax()
    )


class vanilla_model(nn.Module):

    def __init__(self, encoder, decoder=None):
        super(vanilla_model, self).__init__()
        self.encoder = encoder
        # freeze encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False


        # need access to these intermediate encoder steps
        # for the AdaIN computation
        encoder_list = list(encoder.children())
        self.encoder_stage_1 = nn.Sequential(*encoder_list[:4])  # input -> relu1_1
        self.encoder_stage_2 = nn.Sequential(*encoder_list[4:11])  # relu1_1 -> relu2_1
        self.encoder_stage_3 = nn.Sequential(*encoder_list[11:18])  # relu2_1 -> relu3_1
        self.encoder_stage_4 = nn.Sequential(*encoder_list[18:31])  # relu3_1 -> relu4_1

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
    def encode(self, X):
        relu1_1 = self.encoder_stage_1(X)
        relu2_1 = self.encoder_stage_2(relu1_1)
        relu3_1 = self.encoder_stage_3(relu2_1)
        relu4_1 = self.encoder_stage_4(relu3_1)
        return relu1_1, relu2_1, relu3_1, relu4_1

    def decode(self, X): #take in the flattened image of relu4_1 feature map
        return self.decoder(X)

    def forward(self, X):
        X = self.encoder(X)
        output = X.view(-1, 512*4*4)
        # print(relu_4.shape)
        return self.decode(output)


    '''

    def encode(self, X):
        X = X.view(X.size(0), -1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        return X

    def decode(self, X):
        X = F.relu(self.fc3(X))
        X = torch.sigmoid(self.fc4(X))
        return X



    def interpolate_points(self, p1, p2, n_steps=10):
        # interpolate ratios between the points
        ratios = np.linspace(0, 1, num=n_steps)
        # linear interpolate vectors
        vectors = list()
        for ratio in ratios:
            v = (1.0 - ratio) * p1 + ratio * p2
            vectors.append(v)
        return np.asarray(vectors)

    def get_bottleneck(self, X):
        X = X.view(1, -1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        return X
        '''

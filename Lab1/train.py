import datetime
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import autoencoderMLP4Layer
from torchsummary import summary
import matplotlib.pyplot as plt


def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device, save_file=None, plot_file=None):
    print("Training started")
    model.train()
    losses_train = []
    final_loss = 0

    for epoch in range(1, n_epochs + 1):
        print("Epoch", epoch)
        loss_train = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.view(imgs.size(0), -1).to(device=device)  # Flatten the input images
            outputs = model(imgs)
            loss = loss_fn(outputs, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        if save_file is not None:
            torch.save(model.state_dict(), save_file)

        scheduler.step(loss_train)

        losses_train.append(loss_train / len(train_loader))
        final_loss = loss_train / len(train_loader)
        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train / len(train_loader)))
    if plot_file is not None:
        plt.figure(2, figsize=(12, 7))
        plt.clf()
        plt.plot(losses_train, label='train')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc=1)
        plt.savefig(plot_file)
    summary(model, (1, 28 * 28))
    return final_loss


if __name__ == "__main__":
    # Parse command-line arguments
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-z", "--bottleneck",type=int, required=True)
    argParser.add_argument("-s", "--paramFile", required=True)
    argParser.add_argument("-p", "--plotFile", required=True)
    argParser.add_argument("-e", "--num_epochs", type=int, required=True)
    argParser.add_argument("-b", "--batch_size", type=int, required=True)
    argParser.add_argument("-n", "--num_of_steps", type=int, required=False)
    args = argParser.parse_args()

    # Import the dataset
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST('./data/mnist', train=True, download=True, transform=train_transform)
    training_data = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    # Set the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create autoencoder
    autoencoder = autoencoderMLP4Layer(N_bottlenecks=args.bottleneck)
    autoencoder.to(device)
    # Define optimizer and learning rate scheduler
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True, factor=0.1,
                                                     min_lr=1e-4)
    # Train the model
    train(args.num_epochs, optimizer, autoencoder, nn.MSELoss(), training_data, scheduler, device, args.paramFile,
          args.plotFile)




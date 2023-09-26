import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from model import autoencoderMLP4Layer
import train

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    # argParser.add_argument("-R", "--run_type", required=True)
    argParser.add_argument("-z", "--bottleneck", type=int, required=True)
    argParser.add_argument("-s", "--paramFile", required=True)
    argParser.add_argument("-p", "--plotFile", required=True)
    argParser.add_argument("-e", "--num_epochs", type=int, required=True)
    argParser.add_argument("-b", "--batch_size", type=int, required=True)
    args = argParser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device: ", device)
    # Load test data
    training_data = train.load_data(args.batch_size)
    # Create autoencoder, optimizer, scheduler
    learning_rate_arr = [0.01, 0.011, 0.001, 0.002, 0.003, 0.004, 0.005, 0.0001, 0.0002, 0.0003, 0.0004]
    final_losses = []
    for idx, lr in enumerate(learning_rate_arr):
        print("Performing training with the Learning rate of: [{}]".format(lr))
        modelName = 'MLP' + str(idx) + '.8pth'
        plotFile = 'loss.MLP' + str(idx) + '.8.png'
        autoencoder = autoencoderMLP4Layer()
        autoencoder.to(device)

        optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True, factor=0.1,
                                                     min_lr=1e-3)
        # # Load saved encoder
        # autoencoder.load_state_dict(torch.load(args.paramFile))
        model_final_loss = train.train(args.num_epochs, optimizer, autoencoder, nn.MSELoss(), training_data, scheduler, device, modelName,
                                       plotFile, False)

        final_losses.append((model_final_loss, lr))
    try:
        print("Summary of tests and their losses:")
        print("==================================")

        with open("summary2.txt", "w") as fp:
            for idx, loss_lr in enumerate(final_losses):
                textToPrint = '{}   | {}   | {}   |'.format(idx, loss_lr[1], loss_lr[0])

                fp.write(textToPrint)
                fp.write('\n')
                print(textToPrint)
    except Exception as e:
        print("Raising exception: ", e)


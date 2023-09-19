import argparse
import torch
import torch.optim as optim
from model import autoencoderMLP4Layer
import train

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-l", "--paramFile", required=True)
    args = argParser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device: ", device)
    # Load test data
    training_data = train.load_data(training=False)
    # Create autoencoder, optimizer, scheduler
    autoencoder = autoencoderMLP4Layer(N_bottlenecks=32)
    autoencoder.to(device)  # Put model on device that will be used to run commands (cpu or gpu)
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True, factor=0.1,
                                                     min_lr=1e-3)

    # Load saved encoder
    autoencoder.load_state_dict(torch.load(args.paramFile))

    # Step 4: Test the encoder, print out 3 images and their reconstructed matches
    print("Step 4: Testing the autoencoder. 3 Images will be displayed\n "
          "along with their reconstructed matches. \n"
          "*Plot images, saved under 'ELEC475Lab1Part4.png' *\n")

    print("Please close window when ready to move on")
    train.test(autoencoder, training_data, device)

    # Step 5: Image denoising, Print 3 images (original, with noise added, reconstructed output)
    print("Step 5: Image Denoising. 3 Images will be displayed in one column\n"
          "Beside each one will be the same image with noise added, and the \n"
          "reconstructed output through the provided model\n"
          "*Plot images, saved under 'ELEC475Lab1Part5.png' *")

    print("Please close window when ready to move on")

    train.testwithNoise(autoencoder, training_data, device)

    # Step 6: Bottleneck Interpolation. Take 2 images, interpolate between them and print the tensors
    #     #         between and the 2 original images
    print("Step 6: Bottleneck Interpolation. 3 pairs of images will be printed,\n"
          "Along with the results of the images being interpolated together \n"
          "*Plot images, saved under 'ELEC475Lab1Part6.png' *")

    print("Please close window when ready to move on")

    train.bottleneck_interpolation(autoencoder, training_data, 8, device)

    print("Script is complete")



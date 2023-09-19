import datetime
import argparse

import numpy
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Lab1.model import autoencoderMLP4Layer
from torchsummary import summary
import matplotlib.pyplot as plt


def load_data(batch_size=28, training=True):
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST('./data/mnist', train=training, download=True, transform=train_transform)
    training_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    return training_data



def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device, save_file=None, plotFile=None, print_results=False):
    print("Training started")
    model.train()
    losses_train = []
    finalLoss = 0

    for epoch in range(1, n_epochs + 1):
        print("Epoch", epoch)
        loss_train = 0.0

        # for batch in loader:
        #     images, labels = batch
        for imgs, labels in train_loader:
            imgs = imgs.view(imgs.size(0), -1).to(device=device)  # Flatten the input images
            outputs = model(imgs)
            loss = loss_fn(outputs, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        # NK: Possibly delete before submission
        # if print_results and (epoch % 10 == 0):
        #     print("Showing results at epoch, ", epoch)
        #     plt.figure(figsize=(4, 2))
        #     plt.gray()
        #     imgs_to_print = saved_images.detach().numpy()
        #     output_to_print = saved_output.detach().numpy()
        #     for i, img in enumerate(imgs_to_print):
        #         if i >= 4:
        #             break
        #         img = img.reshape(-1, 28, 28)
        #         plt.subplot(2,4, i+1)
        #         plt.imshow(img[0])
        #     for i, output in enumerate(output_to_print):
        #         if i >= 4:
        #             break
        #         output = output.reshape(-1, 28, 28)
        #         plt.subplot(2,4, 4+i+1)
        #         plt.imshow(output[0])
        #     plt.show()

        if save_file is not None:
            torch.save(model.state_dict(), save_file)

        if plotFile is not None:
            plt.figure(2, figsize=(12, 7))
            plt.clf()
            plt.plot(losses_train, label='train')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc=1)
            plt.savefig(plotFile)

        scheduler.step(loss_train)

        losses_train.append(loss_train / len(train_loader))
        finalLoss = loss_train / len(train_loader)
        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train / len(train_loader)))
    summary(model, (1, 28 * 28))
    return finalLoss


def test(model, loader, device, num_of_images=3, save_default=True):

    model.eval()
    tensors_to_print = []

    with torch.no_grad():

        for indx, (imgs, label) in enumerate(loader):
            # load into selected device (cpu or gpu)
            imgs = imgs.to(device=device)
            imgs = imgs.view(imgs.size(0), -1)

            # Forward pass through the autoencoder
            reconstructed_imgs = model(imgs)

            # Display the original and reconstructed images (you can use Matplotlib)
            original_img = imgs.view(-1, 28, 28).cpu().numpy()
            reconstructed_img = reconstructed_imgs.view(-1, 28, 28).cpu().numpy()

            tensors_to_print.append((original_img[0], reconstructed_img[0]))

        plt.figure(figsize=(8, 8))
        counter = 1
        for idx, images in enumerate(tensors_to_print):
            if idx >= num_of_images:
                break
            original = images[0]
            new_image = images[1]
            plt.subplot(num_of_images, 2, counter)
            if idx == 0:
                plt.title("Original Image")
            plt.imshow(original, cmap="gray")

            plt.subplot(num_of_images, 2, counter+1)
            if idx == 0:
                plt.title("Reconstructed Image")
            plt.imshow(new_image, cmap="gray")
            counter += 2

        plt.show()

        if save_default:
            plt.savefig("ELEC475Lab1Part4.png")


def testwithNoise(model, loader, device, num_of_images=3, save_default=True):
    images_to_print = []
    model.eval()

    with torch.no_grad():

        for imgs, label in loader:

            noise = torch.randn_like(imgs) * 0.2 #0.2 to reduce noise
            noisyImg = imgs + noise

            noisyImg = noisyImg.to(device=device)
            noisyImg = noisyImg.view(imgs.size(0), -1)
            # Forward pass through the autoencoder
            reconstructed_imgs = model(noisyImg)

            # Display the original and reconstructed images (you can use Matplotlib)
            original_img = imgs.view(-1, 28, 28).cpu().numpy()
            img_w_noise = noisyImg.view(-1, 28, 28).cpu().numpy()
            reconstructed_img = reconstructed_imgs.view(-1, 28, 28).cpu().numpy()
            images_to_print.append((original_img[0], img_w_noise[0], reconstructed_img[0]))
        counter = 1
        plt.figure(figsize=(8, 5))

        for index, element in enumerate(images_to_print):
            if index >= num_of_images:
                break
            plt.subplot(num_of_images, 3, counter)
            if index == 0:
                plt.title("Original Image")
            plt.imshow(element[0], cmap="gray")

            plt.subplot(num_of_images, 3, counter+1)
            if index == 0:
                plt.title("Noisy Image")
            plt.imshow(element[1], cmap="gray")

            plt.subplot(num_of_images, 3, counter+2)
            if index == 0:
                plt.title("Reconstructed Image")
            plt.imshow(element[2], cmap="gray")
            counter += 3
        plt.show()
        if save_default:
            plt.savefig("ELEC475Lab1Part5.png")


def interpolateImages(model, tensor1, tensor2, steps):

    tensor_arr = []
    for x in numpy.linspace(0, 1, steps):
        temp_tensor = x * tensor1 + (1 - x) * tensor2
        tensor_arr.append(temp_tensor)

    return tensor_arr


def bottleneck_interpolation(model, train_loader, steps, device, num_of_images=3, save_default=True):
    model.eval()
    tensors_to_print = []

    with torch.no_grad():

        for batch_idx, (images, label) in enumerate(train_loader):
            images = images.view(images.size(0), -1).to(device=device)  # Flatten the input images
            # Store 2 images to be interpolated
            tensor1 = model.encoder(images[0])
            tensor2 = model.encoder(images[1])
            tensor_arr = interpolateImages(model, tensor2, tensor1, steps)

            images = images.view(-1, 28, 28).cpu().numpy()
            image1 = images[0]
            image2 = images[1]

            tensors_to_print.append((image1, image2, tensor_arr))
        plt.figure(figsize=(8, 8))
        counter = 1
        for printIndx, tensor_element in enumerate(tensors_to_print):
            if printIndx >= num_of_images:
                break
            image1 = tensor_element[0]
            image2 = tensor_element[1]
            tensor_array = tensor_element[2]

            plt.subplot(num_of_images, steps + 2, counter)
            if printIndx == 0:
                plt.title("Img 1")
            plt.imshow(image1, cmap="gray")

            counter += 1

            for idx, tensor in enumerate(tensor_array):
                tensor_output = model.decoder(tensor)
                temp_tensor = tensor_output.view(28, 28).cpu().numpy()
                plt.subplot(num_of_images, steps + 2, counter)
                if printIndx == 0:
                    plt.title("Step{}".format(idx + 1))
                plt.imshow(temp_tensor, cmap="gray")
                counter += 1

            plt.subplot(num_of_images, steps + 2, counter)
            if printIndx == 0:
                plt.title("Img 2")
            plt.imshow(image2, cmap="gray")
            counter += 1

        plt.show()
        if save_default:
            plt.savefig("ELEC475Lab1Part6.png")



if __name__ == "__main__":
    # Parse command-line arguments
    argParser = argparse.ArgumentParser()
    # argParser.add_argument("-R", "--run_type", required=True)
    argParser.add_argument("-z", "--bottleneck",type=int, required=True)
    argParser.add_argument("-s", "--paramFile", required=True)
    argParser.add_argument("-p", "--plotFile", required=True)
    argParser.add_argument("-e", "--num_epochs", type=int, required=True)
    argParser.add_argument("-b", "--batch_size", type=int, required=True)
    argParser.add_argument("-n", "--num_of_steps", type=int, required=False)
    args = argParser.parse_args()

    # Set the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Import the dataset
    training_data = load_data(args.batch_size)
    # Create autoencoder
    autoencoder = autoencoderMLP4Layer(N_bottlenecks=args.bottleneck)
    autoencoder.to(device)
    # Define optimizer and learning rate scheduler
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1e-3, last_epoch=-1, verbose=True )

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True, factor=0.1,
    #                                                  min_lr=1e-4)
    # Train the model
    train(args.num_epochs, optimizer, autoencoder, nn.MSELoss(), training_data, scheduler, device, args.paramFile,
          args.plotFile)

    # if args.run_type == 'train':
        # elif args.run_type == 'denoise':
        #     print("Running Denoise data")
        #     autoencoder.load_state_dict(torch.load(args.paramFile))
        #     testwithNoise(autoencoder, training_data, device)
        # elif args.run_type == 'bottleneck':
        #     print("Running bottleneck interpolation")
        #     autoencoder.load_state_dict(torch.load(args.paramFile))
        #     if args.num_of_steps is not None:
        #         bottleneck_interpolation(autoencoder, training_data, args.num_of_steps, device)
        #     else:
        #         bottleneck_interpolation(autoencoder, training_data, 8, device)
        # else:
        #     autoencoder.load_state_dict(torch.load(args.paramFile))
        #     test(autoencoder, training_data, device)



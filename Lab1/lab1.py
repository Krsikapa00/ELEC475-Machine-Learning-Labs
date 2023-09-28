import argparse
import numpy
import torch
from model import autoencoderMLP4Layer
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def load_data(batch_size=28, training=True):
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST('./data/mnist', train=training, download=True, transform=train_transform)
    training_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    return training_data


def run_autoencoder(model, device, original):
    original = original.to(device=device)
    original = original.view(original.size(0), -1)
    # Forward pass through the autoencoder
    reconstructed_imgs = model(original)
    original_img = original.view(-1, 28, 28).cpu().numpy()
    reconstructed_img = reconstructed_imgs.view(-1, 28, 28).cpu().numpy()

    return original_img, reconstructed_img, original


def test(model, loader, device, num_of_images=3):
    model.eval()
    tensors_to_print = []

    with torch.no_grad():

        for indx, (imgs, label) in enumerate(loader):
            # load into selected device (cpu or gpu)
            original_img, reconstructed_img, imgs = run_autoencoder(model, device, imgs)

            tensors_to_print.append((original_img[0], reconstructed_img[0]))

        plt.figure(num=1, figsize=(8, 8))
        plt.clf()

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
            plt.subplot(num_of_images, 2, counter + 1)
            if idx == 0:
                plt.title("Reconstructed Image")
            plt.imshow(new_image, cmap="gray")
            counter += 2

        plt.show()


def testwithnoise(model, loader, device, num_of_images=3):
    images_to_print = []
    model.eval()

    with torch.no_grad():

        for imgs, label in loader:
            noise = torch.randn_like(imgs) * 0.3  # 0.2 to reduce noise
            noisyImg = imgs + noise

            img_w_noise, reconstructed_img, noisyImg = run_autoencoder(model, device, noisyImg)

            # Display the original and reconstructed images (you can use Matplotlib)
            original_img = imgs.view(-1, 28, 28).cpu().numpy()
            images_to_print.append((original_img[0], img_w_noise[0], reconstructed_img[0]))
        counter = 1
        plt.figure(num=1, figsize=(8, 5))
        plt.clf()

        for index, element in enumerate(images_to_print):
            if index >= num_of_images:
                break
            plt.subplot(num_of_images, 3, counter)
            if index == 0:
                plt.title("Original Image")
            plt.imshow(element[0], cmap="gray")

            plt.subplot(num_of_images, 3, counter + 1)
            if index == 0:
                plt.title("Noisy Image")
            plt.imshow(element[1], cmap="gray")

            plt.subplot(num_of_images, 3, counter + 2)
            if index == 0:
                plt.title("Reconstructed Image")
            plt.imshow(element[2], cmap="gray")
            counter += 3

        plt.show()


def interpolateimages(tensor1, tensor2, steps):
    tensor_arr = []
    for x in numpy.linspace(0, 1, steps):
        temp_tensor = x * tensor1 + (1 - x) * tensor2
        tensor_arr.append(temp_tensor)

    return tensor_arr


def bottleneck_interpolation(model, train_loader, steps, device, num_of_images=3):
    model.eval()
    tensors_to_print = []

    with torch.no_grad():

        for batch_idx, (images, label) in enumerate(train_loader):
            images = images.view(images.size(0), -1).to(device=device)  # Flatten the input images
            # Store 2 images to be interpolated
            tensor1 = model.encoder(images[0])
            tensor2 = model.encoder(images[1])
            tensor_arr = interpolateimages(tensor2, tensor1, steps)

            images = images.view(-1, 28, 28).cpu().numpy()
            image1 = images[0]
            image2 = images[1]

            tensors_to_print.append((image1, image2, tensor_arr))
        plt.figure(num=1, figsize=(8, 4))
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


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-l", "--paramFile", required=True)
    args = argParser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device: ", device)
    print("Run file with -s argument to save images on each step")
    # Load test data
    training_data = load_data(training=False)
    # Create autoencoder and load saved model from provided argument
    autoencoder = autoencoderMLP4Layer()
    autoencoder.to(device)  # Put model on device that will be used to run commands (cpu or gpu)
    autoencoder.load_state_dict(torch.load(args.paramFile))

    # Step 4: Test the encoder, print out 3 images and their reconstructed matches
    print("Step 4: Testing the autoencoder. 3 Images will be displayed\n "
          "along with their reconstructed matches. \n"
          "*Plot images, saved under 'ELEC475Lab1Part4.png' *")

    print("Please close window when ready to move on\n\n")
    test(autoencoder, training_data, device)

    # Step 5: Image denoising, Print 3 images (original, with noise added, reconstructed output)
    print("Step 5: Image Denoising. 3 Images will be displayed in one column\n"
          "Beside each one will be the same image with noise added, and the \n"
          "reconstructed output through the provided model\n"
          "*Plot images, saved under 'ELEC475Lab1Part5.png' *")

    print("Please close window when ready to move on\n\n")

    testwithnoise(autoencoder, training_data, device)

    # Step 6: Bottleneck Interpolation. Take 2 images, interpolate between them and print the tensors
    #     #         between and the 2 original images
    print("Step 6: Bottleneck Interpolation. 3 pairs of images will be printed,\n"
          "Along with the results of the images being interpolated together \n"
          "*Plot images, saved under 'ELEC475Lab1Part6.png' *")

    print("Please close window when ready to move on\n\n")

    bottleneck_interpolation(autoencoder, training_data, 8, device)

    print("Script is complete")

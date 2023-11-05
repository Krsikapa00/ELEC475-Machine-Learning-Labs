import argparse
import os
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import vanilla_model as vanilla
import train
from torch.utils.data import DataLoader, Dataset
import time as t


def input_num():
    num = int(input('Input a number between 0 and 10: '))
    while True:
        if num > 10 or num < 0:
            num = int(input('Out of Bounds. Input a number between 0 and 10: '))
        else:
            return num


def top1_error(predictions, labels):
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    top1_err = 1 - correct / len(labels)
    return top1_err


def top5_error(predictions, labels):
    _, top5_predictions = torch.topk(predictions, 5, 1, largest=True, sorted=True)
    correct = 0
    for i in range(len(labels)):
        if labels[i] in top5_predictions[i]:
            correct += 1
    top5_err = 1 - correct / len(labels)
    return top5_err


if __name__ == '__main__':

    image_size = 512
    device = 'cpu'

    parser = argparse.ArgumentParser()

    parser.add_argument('--encoder_file', type=str, help='encoder weight file')
    parser.add_argument('--decoder_file', type=str, help='decoder weight file')
    parser.add_argument('--cuda', type=str, help='[y/N]')

    opt = parser.parse_args()

    train_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    cifar_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    cifar_data = DataLoader(cifar_dataset, batch_size=10000, shuffle=True)

    # output_format = opt.content_image[opt.content_image.find('.'):]

    decoder_file = opt.decoder_file
    encoder_file = opt.encoder_file

    use_cuda = False
    if opt.cuda == 'y' or opt.cuda == 'Y':
        use_cuda = True
    out_dir = './output/'
    os.makedirs(out_dir, exist_ok=True)

    '''
	if torch.cuda.is_available() and use_cuda:
		print('using cuda ...')
		model.cuda()
		input_tensor = input_tensor.cuda()
	else:
		print('using cpu ...')
'''
    if torch.cuda.is_available() and opt.cuda == 'Y':
        if torch.cuda.device_count() > 1:
            device = torch.device("cuda:1")
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print("Using device: {}".format(device))

    encoder = vanilla.encoder_decoder.encoder
    encoder.load_state_dict(torch.load(encoder_file, map_location='cpu'))
    frontend = vanilla.encoder_decoder.frontend
    frontend.load_state_dict(torch.load(decoder_file, map_location='cpu'))
    model = vanilla.vanilla_model(encoder, frontend)

    total_losses = []
    total_top1_accuracy = []
    total_top5_accuracy = []
    final_loss = 0.0

    model.to(device=device)
    model.eval()

    print('model loaded OK!')

    #	total_top1_accuracy.append(float(top_1_acc))
    #	total_top5_accuracy.append(float(top_5_acc))
    # print("Accuracy= TOP-1:   {}  |  TOP-5:    {}".format(top_1_acc, top_5_acc))

    # content_image = transforms.Resize(size=image_size)(content_image)

    # input_tensor = transforms.ToTensor()(content_image).unsqueeze(0)

    print("Using device: {}".format(device))

    out_tensor = None
    with torch.no_grad():
        top1_count = 0
        top5_count = 0

        for idx, data in enumerate(cifar_data):
            t_3 = t.time()
            imgs, labels = data[0].to(device), data[1].to(device)
            out_tensor = model(imgs)

            _, top1_predicted = torch.max(out_tensor, 1)
            _, top5_predictions = torch.topk(out_tensor, 5, 1, largest=True, sorted=True)

            # if labels[idx] == top1_predicted:
            #	top1_count = top1_count + 1

            # if label[idx] in top5_predictions:
            #	top5_count = top5_count + 1

            top1_count += torch.sum(labels == top1_predicted).item()

            for i in range(len(labels)):
                if labels[i] in top5_predictions[i]:
                    top5_count += 1

            print('Image #{}/{}         Time: {}'.format(idx + 1, len(cifar_data), (t.time() - t_3)))

    top1_err = 1 - top1_count / len(cifar_dataset)
    top5_err = 1 - top5_count / len(cifar_dataset)

    print(f"Top-1 Error Rate: {top1_err * 100}%")
    print(f"Top-5 Error Rate: {top5_err * 100}%")

    # filename = os.path.join(os.path.abspath(folder), 'output_results')

    '''
	print("Length: {}    {}   {}".format(total_top5_accuracy, total_top1_accuracy, total_losses))
	try:
		with1111111 open(filename, 'w') as file:
			file.write("Accuracy & loss results per Epoch: (Top 1,    Top 5,   Loss)\n")
			for i in range(n_epochs):
				top_1_item = total_top1_accuracy[i]
				top_5_item = total_top5_accuracy[i]
				loss = total_losses[i]
				file.write("Epoch {}:  ({}   ,{},    {})\n".format(i + 1, top_1_item, top_5_item, loss))
			file.close()
	except Exception as e:
		print(f"An error occurred while loading arrays: {str(e)}")


	save_file = out_dir     + opt.decoder_file[opt.decoder_file.rfind('_')+1: opt.decoder_file.find('.')] + "/" \
							+ opt.content_image[opt.content_image.rfind('/')+1: opt.content_image.find('.')] \
							+"_style_"+ opt.style_image[opt.style_image.rfind('/')+1: opt.style_image.find('.')] \
							+ "_alpha_" + str(alpha) \
							+ output_format

	print('saving output file: ', save_file)
	save_image(out_tensor, save_file)'''
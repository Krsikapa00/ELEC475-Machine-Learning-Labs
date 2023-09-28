import argparse
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import AdaIN_net as net

if __name__ == '__main__':

	image_size = 512
	device = 'cpu'

	parser = argparse.ArgumentParser()
	# parser.add_argument('-content_dir', type=str, help='test image')
	# parser.add_argument('-style_image', type=str, help='style image')
	# parser.add_argument('-encoder_file', type=str, help='encoder weight file')
	# parser.add_argument('-decoder_file', type=str, help='decoder weight file')
	# parser.add_argument('-alpha', type=float, default=1.0, help='Level of style transfer, value between 0 and 1')
	# parser.add_argument('-cuda', type=str, help='[y/N]')

	parser.add_argument('--content_dir', type=str, required=True,
						help='Directory path to a batch of content images')
	parser.add_argument('--style_dir', type=str, required=True,
						help='Directory path to a batch of style images')

	# training options
	parser.add_argument('--gamma', default=1.0,
						help='Gamma value')
	parser.add_argument('--e', type=int, default=50)
	parser.add_argument('--b', type=int, default=8)
	parser.add_argument('--l', type=int, help="Encoder.pth")
	parser.add_argument('--s', type=int, help="Decoder.pth")
	parser.add_argument('--p', type=int, help="decoder.png")
	parser.add_argument('--cuda', type=int, default='Y')
	# python3 train.py
	# 	-content_dir. /../../../ datasets / COCO100 /
	# 	-style_dir. /../../../ datasets / wikiart100 /
	# 	-gamma 1.0
	# 	-e 20
	# 	-b 20
	# 	-l encoder.pth
	# 	-s decoder.pth
	# 	-p decoder.png
	# 	-cuda Y


	opt = parser.parse_args()
	content_image = Image.open(opt.content_image)
	style_image = Image.open(opt.style_image)
	output_format = opt.content_image[opt.content_image.find('.'):]
	decoder_file = opt.decoder_file
	encoder_file = opt.encoder_file
	alpha = opt.alpha
	use_cuda = False
	if opt.cuda == 'y' or opt.cuda == 'Y':
		use_cuda = True
	out_dir = './output/'
	os.makedirs(out_dir, exist_ok=True)

	encoder = net.encoder_decoder.encoder
	encoder.load_state_dict(torch.load(encoder_file, map_location='cpu'))
	decoder = net.encoder_decoder.decoder
	decoder.load_state_dict(torch.load(decoder_file, map_location='cpu'))
	model = net.AdaIN_net(encoder, decoder)

	model.to(device=device)
	model.eval()

	print('model loaded OK!')

	content_image = transforms.Resize(size=image_size)(content_image)
	style_image = transforms.Resize(size=image_size)(style_image)

	input_tensor = transforms.ToTensor()(content_image).unsqueeze(0)
	style_tensor = transforms.ToTensor()(style_image).unsqueeze(0)

	if torch.cuda.is_available() and use_cuda:
		print('using cuda ...')
		model.cuda()
		input_tensor = input_tensor.cuda()
		style_tensor = style_tensor.cuda()
	else:
		print('using cpu ...')

	out_tensor = None
	with torch.no_grad():
		out_tensor = model(input_tensor, style_tensor, alpha)

	save_file = out_dir + opt.content_image[opt.content_image.rfind('/')+1: opt.content_image.find('.')] \
							+"_style_"+ opt.style_image[opt.style_image.rfind('/')+1: opt.style_image.find('.')] \
							+ "_alpha_" + str(alpha) \
							+ output_format
	print('saving output file: ', save_file)
	save_image(out_tensor, save_file)
import torch
from argparse import ArgumentParser


# Device on which variables and model will be placed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Configuration object with contains every parameter necessary
# for the process of loading/processing images and synthetization process
parser = ArgumentParser()
# Image loading parameters
parser.add_argument("--content_loc", type=str,
                    default='images/content/green_bridge.jpg', help='location of the content image')
parser.add_argument("--style_loc", type=str,
                    default='images/style/vg_la_cafe.jpg', help='location of the style image')
parser.add_argument("--output_folder", type=str, default="output",
                    help='output folder for generated images')
parser.add_argument("--img_h", type=int, default=512,
                    help='width of the image')
parser.add_argument("--img_w", type=int, default=512,
                    help='height of the image')

# Parameters necessary for synthetization process
parser.add_argument("--iterations", type=int, default=500,
                    help='number of iterations to perform')
parser.add_argument("--optimizer", type=str, choices=['lbfgs', 'adam'],
                    default='lbfgs', help='NST optimizer')
parser.add_argument("--lr", type=float, default=1e-2,
                    help="learning rate for Adam optimizer")
parser.add_argument("--init_mode", type=str, choices=['content', 'random'],
                    default='content',
                    help="intialization mode for generated image")
parser.add_argument("--content_w", type=float, default=1,
                    help="alpha parameter, the weight of content loss")
parser.add_argument("--style_w", type=float, default=1e6,
                    help='beta parameter, the weight of style loss')

config = parser.parse_args(args=[])


# INDICES OF USED LAYERS IN VGG19 NETWORK
# Indices of layers from which we extract style features
STYLE_LAYERS = [1, 6, 11, 20, 29]
# Indices of layers from which we extract content features
CONTENT_LAYERS = [22]

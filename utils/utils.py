from utils.globals import *

import torchvision.transforms as transforms
import torchvision.models as models
from torch.optim import Adam, LBFGS

from PIL import Image
import matplotlib.pyplot as plt


def load_model(device):
    ''' Loads feature extractor from the
        VGG19 model and ports it to the
        specified device.

    Arguments:
     device (torch.device): device on which we
                            wish to port the model.
    Returns:
     neural_net (nn.Module): feature extractor of the
                             Vgg19 network
    '''
    neural_net = models.vgg19(pretrained=True, progress=True).features
    neural_net = neural_net.eval()

    for param in neural_net.parameters():
        param.requires_grad = False

    return neural_net.to(device)


def gram_matrix(tensor):
    ''' Calculates Gram matrix of a
        tensor. Gram matrix is used
        later for capturing the style
        at some layers.

    Arguments:
        tensor (torch.tensor): a set of features
                               for which we calculate
                               the Gram matrix
    Returns:
        G (torch.tensor): Gram matrix of specified
                          features
    '''
    bat, ch, h, w = tensor.shape

    tensor = tensor.view(bat * ch, h * w)
    Gram = tensor.mm(tensor.t())

    return Gram.div(bat * ch * w * h)


def get_optimizer(generated_img, config):
    ''' Acquires wished optimizer. We choose
        between L-BFGS & Adam.

    Arguments:
        generated_img (torch.Tensor): generated image
        config (ArgumentParser): object which contains
                                 all hyperparameters
    Returns:
        optimizer: an optimizer we have chosen
    '''
    if config.optimizer == 'lbfgs':
        print("Using L-BFGS optimizer.")
        return LBFGS([generated_img.requires_grad_()])
    else:
        print(f"Using Adam optimizer with learning rate:{config.lr}.")
        return Adam((generated_img,), lr=config.lr)


def initialize_image(config, device, content, style):
    ''' Initializes image. Image can be initialized to
        be the same as content or style image, or we 
        can initialize to randomly with normal distribution

    Arguments:
        config (ArgumentParser): object which contains
                                 all hyperparameters
        device (torch.device): device on which we port the
                               initial generated image
        content (torch.Tensor): loaded content_image
        style (torch.Tensor): loaded style_image
    Returns:
        generated_image (torch.Tensor): initialized image
    '''
    if config.init_mode == 'random':
        generated_image = torch.randn(content.data.size(), device)
    elif config.init_mode == 'content':
        generated_image = (content.cpu()).clone().to(device)
    else:
        generated_image = (style.cpu()).clone().to(device)

    generated_image.requires_grad = True
    return generated_image


def load_image(img_path, config, device):
    ''' Loades the image from specified path
        as a PyTorch tensor and resizes it to the
        size specified in the config arguments.
        Image gets transfered to the desired device.
        Batch dimension gets added to the image.

    Arguments:
        img_path (string): path to the image we wish
                           to load
        config (ArgumentParser): object which contains
                                 all hyperparameters
        device (torch.device): a device on which we 
                            put all variables and model
    Returns:
        img (torch.Tensor): loaded image, ported to device
    '''
    # Define all necessary transforms we need to perform
    # on the specified image as a loader object
    ImageLoader = transforms.Compose([
        transforms.Resize([config.img_h, config.img_w]),
        transforms.ToTensor()
    ])

    # Open image at specified path (we convert it
    # to RGB because .png images sometimes load
    # with 4 channels, we reduce it to 3 this way)
    img = Image.open(img_path).convert('RGB')
    # Perform necessary transforms
    img = ImageLoader(img).unsqueeze(0)
    # We don't want to modify loaded images
    # so we exclude them from computation graph
    img.requires_grad = False

    return img.to(device, torch.float)


def show_image(img, title=None):
    ''' Displays image on to a figure.

    Arguments:
        img (torch.Tensor): an image to show
        title (string): title of the figure
    '''
    # Define a transformation of Tensor to PIL image
    ImageDeloader = transforms.ToPILImage()

    # Remove batch dimension, we can't show a 4D
    # image on a matplotlib figure
    img = img.squeeze(0)
    # Transform the image to a PIL image
    img = ImageDeloader(img.cpu().clone())

    # Show the image
    plt.ioff()
    plt.figure()
    plt.imshow(img)
    plt.axis('off')

    # Display the title (if any)
    if title:
        plt.title(title)

    plt.show()
    plt.pause(0.0001)

from utils.globals import *
from model import Vgg19NST
from utils.utils import initialize_image, get_optimizer

from torch.optim import Adam, LBFGS

import time


def Adam_NST(neural_net, generated_img, optimizer, style_losses, content_losses):
    ''' Performs the Neural Style Transfer using Adam optimizer.

    Arguments:
        neural_net (nn.Module): neural network used for inference
        generated_img (torch.Tensor): initialized image
        optimizer: Adam optimizer
        style_losses (list): MSE losses for style Gram matrices
        content_losses (list): MSE losses for content features
    Returns:
        generated_img (torch.Tensor): synthetized image(content+style)
    '''
    for i in range(config.iterations):
        # Perform pixel value clipping to the generated image
        generated_img.data.clamp_(0, 1)
        optimizer.zero_grad()
        # Propagate current version of image through the ne
        neural_net(generated_img)

        style_loss = 0
        content_loss = 0

        for sl in style_losses:
            style_loss += sl.loss
        for cl in content_losses:
            content_loss += cl.loss

        # Apply weighting for both losses
        # (as proposed in the papers)
        style_loss *= config.style_w
        content_loss *= config.content_w

        loss = style_loss + content_loss
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"Iter: {i}")
            print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                style_loss.item(), content_loss.item()))
            print()

    return generated_img


def LBFGS_NST(neural_net, generated_img, optimizer, style_losses, content_losses):
    ''' Performs the Neural Style Transfer using L-BFGS optimizer.

    Arguments:
        neural_net (nn.Module): neural network used for inference
        generated_img (torch.Tensor): initialized image
        optimizer: L-BFGS optimizer
        style_losses (list): MSE losses for style Gram matrices
        content_losses (list): MSE losses for content features
    Returns:
        generated_img (torch.Tensor): synthetized image(content+style)
    '''
    cnt = 0
    while cnt <= config.iterations:
        def closure():
            ''' Closure function for L-BFGS optimizer. '''
            nonlocal cnt
            # Perform pixel value clipping to the generated image
            generated_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            # Propagate current version of image through the ne
            neural_net(generated_img)

            style_loss = 0
            content_loss = 0

            for sl in style_losses:
                style_loss += sl.loss
            for cl in content_losses:
                content_loss += cl.loss

            # Apply weighting for both losses
            # (as proposed in the papers)
            style_loss *= (config.style_w / len(style_losses))
            content_loss *= config.content_w

            loss = style_loss + content_loss
            loss.backward()
            cnt += 1
            if cnt % 50 == 0:
                print(f"Iter: {cnt}")
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_loss.item(), content_loss.item()))
                print()

            return style_loss + content_loss

        optimizer.step(closure)

    return generated_img


def NST(content_img, style_img, config, device):
    ''' Performs the Neural Style Transfer by calling
        the appropriate function (Adam/L-BFGS) based 
        on config's optimizer argument.

    Arguments:
        content_img (torch.Tensor): original content image
        style_img (torch.Tensor): original style image
        config (ArgumentParser): object which contains
                                 all hyperparameters
        device (torch.device): device on which we place
                               model and variables
    Returns:
        generated_img (torch.tensor): synthetized image
    '''
    # Initialize the NST model
    neural_net = Vgg19NST(content_img, style_img, device)
    # Acquire the loss-layer objects
    style_losses, content_losses = neural_net.get_losses()

    # Initialize the synthetized image
    generated_img = initialize_image(config, device, content_img, style_img)
    # Acquire the optimizer
    optimizer = get_optimizer(generated_img, config)

    print('Synthetization process started..')
    start = time.time()

    # Perform the Neural Style Transfer
    if isinstance(optimizer, Adam):
        generated_img = Adam_NST(neural_net, generated_img,
                                 optimizer, style_losses, content_losses)
    else:
        generated_img = LBFGS_NST(
            neural_net, generated_img, optimizer, style_losses, content_losses)

    # Perform the final clipping
    generated_img.data.clamp_(0, 1)
    print(
        f"Total synthetization time is: {round(time.time()-start, 1)} seconds.\n")

    return generated_img

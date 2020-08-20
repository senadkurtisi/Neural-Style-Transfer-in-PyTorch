from utils.globals import *
from utils.utils import load_model, gram_matrix

import torch.nn.functional as F
import torch.nn as nn

import copy


class ContentLoss(nn.Module):
    ''' A layer in which we calculate the MSE
        difference between target content features
        and current content features of generated
        image.
    '''

    def __init__(self, target_content_features):
        super().__init__()
        # Memorizing the content features of the
        # original content image
        self.target_content_features = target_content_features

    def forward(self, x):
        # calculating the MSE difference between content
        # features of original content and generated image
        self.loss = F.mse_loss(self.target_content_features, x)
        # last layers output is just propagated since this
        # layer only calculates MSE difference(loss)
        return x


class StyleLoss(nn.Module):
    ''' A layer in which we calculate the MSE
        difference between Gram matrix of target
        style features and current style features
        of generated image.
    '''

    def __init__(self, target_style_features):
        super().__init__()
        # Memorizing the Gram matrix for the
        # target style features
        self.target_style_gram = gram_matrix(target_style_features).detach()

    def forward(self, x):
        # Calculating the Gram matrix for the given features
        G = gram_matrix(x)
        # Calculating the MSE difference between target Gram
        # matrix and Gram of given features
        self.loss = F.mse_loss(self.target_style_gram, G)
        # last layers output is just propagated since this
        # layer only calculates MSE difference(loss)
        return x


class NormRGB(nn.Module):
    ''' Layer which performs normalization of
        te input RGB image to mean and std of
        ImageNet dataset (Vgg19 was trained on
        this dataset). Input images need to be
        in the [0,1] range.
    '''

    def __init__(self, device):
        super().__init__()
        # ImageNet dataset metrics
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(device)

        self.mean = self.mean.view(-1, 1, 1)
        self.std = self.std.view(-1, 1, 1)

    def forward(self, x):
        # Propagating the normalized image
        return (x - self.mean) / self.std


class Vgg19NST(nn.Module):
    ''' Model which contains only layers from Vgg19
        which are necessary for NST. We include only
        layers up to conv5_1 as proposed in the original
        NST paper: https://arxiv.org/pdf/1508.06576.pdf
    '''

    def __init__(self, content_img, style_img, device):
        super().__init__()
        # Load the Vgg19 feature extractor
        neural_net = copy.deepcopy(load_model(device))
        # Create the first layer of the NST network (norm)
        norm_layer = NormRGB(device)

        self.content_img = content_img
        self.style_img = style_img

        # MSE losses at adequate layers
        self.content_losses = []
        self.style_losses = []

        # Borders define positions at which we place
        # ContentLoss and StyleLoss layers
        self.borders = sorted(STYLE_LAYERS.copy()
                              + CONTENT_LAYERS.copy())

        # Initialize the model with first layer
        # being the normalization layer
        self.model = nn.Sequential(norm_layer)

        # Complete the model
        self.populate_blocks(neural_net)

    def get_losses(self):
        ''' Returns the loss-layer objects. '''
        return self.style_losses, self.content_losses

    def populate_blocks(self, neural_net):
        ''' Adds remaining layers of the Vgg19 feature
            extractor to the NST neural network.
        '''
        block_index = 1  # used for name forming

        for cnt, layer in enumerate(neural_net.children()):
            # Name base, defined by the type of the layer
            if isinstance(layer, nn.ReLU):
                layer = nn.ReLU(inplace=False)
                name = 'relu'
            elif isinstance(layer, nn.Conv2d):
                name = 'conv'
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool'
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn'

            name += str(cnt)
            # Add the layer to our NST model
            self.model.add_module(name, layer)

            if cnt in STYLE_LAYERS:  # Do we need to place StyleLoss layer?
                target = self.model(self.style_img)
                style_layer = StyleLoss(target)
                name = 'StyleLoss_{}'.format(min(block_index, 5))

                self.model.add_module(name, style_layer)
                self.style_losses.append(style_layer)
            elif cnt in CONTENT_LAYERS:  # Do we need to place ContentLoss layer?
                target = self.model(self.content_img)
                content_layer = ContentLoss(target)
                name = 'ContentLoss'

                self.model.add_module(name, content_layer)
                self.content_losses.append(content_layer)

            if cnt == self.borders[-1]:
                # We only include layers up to conv5_1
                # As proposed in the original NST paper:
                # https://arxiv.org/pdf/1508.06576.pdf
                return
            elif cnt == self.borders[block_index - 1]:
                block_index += 1

    def forward(self, input):
        return self.model(input)

import torch
import torch.nn as nn

def conv2d_block(in_channels, out_channels, kernel_size, padding, bn, act, init_layers):
    # Define the convolutional layer
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)]

    # Add batch normalization if 'bn' is True
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))

    # Add the activation function
    if act == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif act == 'lrelu':
        layers.append(nn.LeakyReLU(0.1, inplace=True))

    # Return the layers as a sequential model
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, num_filters, use_bn=True, init_layers=True):
        super(ResidualBlock, self).__init__()

        # First convolutional block with 1x1 kernel
        self.conv1 = conv2d_block(num_filters, num_filters//2, 1, padding=0,
                                  bn=use_bn, act='relu', init_layers=init_layers)

        # Second convolutional block with 3x3 kernel
        self.conv2 = conv2d_block(num_filters//2, num_filters//2, 3, padding=1,
                                  bn=use_bn, act='relu', init_layers=init_layers)

        # Third convolutional block with 1x1 kernel and no activation function
        self.conv3 = conv2d_block(num_filters//2, num_filters, 1, padding=0,
                                  bn=use_bn, act=None, init_layers=init_layers)

    def forward(self, x):
        # Pass the input through the convolutional blocks
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        x_out = self.conv3(x_out)

        # Add the input to the output of the convolutional blocks
        return x + x_out

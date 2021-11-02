import math
import torch
import torch.nn as nn
from torch.nn.functional import conv1d
from torch.nn.functional import conv2d
from torch.nn.functional import conv3d
from torch.nn.functional import conv_transpose1d
from torch.nn.functional import conv_transpose2d
from torch.nn.functional import conv_transpose3d
import scipy.io
import numpy as np
import os

from utils.utils import ShiftedReLU


class ConvSparseLayer(nn.Module):
    """
    An implementation of a Convolutional Sparse Layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, lam=0.5, activation_lr=1e-1,
                 max_activation_iter=200, rectifier=True, convo_dim=2):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_dim = convo_dim

        if isinstance(kernel_size, int):
            self.kernel_size = self.conv_dim * (kernel_size,)
        else:
            self.kernel_size = kernel_size

        if isinstance(stride, int):
            self.stride = self.conv_dim * (stride,)
        else:
            self.stride = stride

        if isinstance(padding, int):
            self.padding = self.conv_dim * (padding,)
        else:
            self.padding = padding

        self.activation_lr = activation_lr
        self.max_activation_iter = max_activation_iter

        self.filters = nn.Parameter(torch.rand((out_channels, in_channels) +
                                               self.kernel_size),
                                    requires_grad=True)
        torch.nn.init.xavier_uniform_(self.filters)
        self.normalize_weights()

        if rectifier:
            self.threshold = ShiftedReLU(lam)
        else:
            self.threshold = nn.Softshrink(lam)

        if self.conv_dim == 1:
            self.convo = conv1d
            self.deconvo = conv_transpose1d
        elif self.conv_dim == 2:
            self.convo = conv2d
            self.deconvo = conv_transpose2d
        elif self.conv_dim == 3:
            self.convo = conv3d
            self.deconvo = conv_transpose3d
        else:
            raise ValueError("Conv_dim must be 1, 2, or 3")

        self.lam = lam

    def normalize_weights(self):
        with torch.no_grad():
            norms = torch.norm(self.filters.reshape(
                self.out_channels, self.in_channels, -1), dim=2, keepdim=True)
            norms = torch.max(norms, 1e-12*torch.ones_like(norms)).view(
                (self.out_channels, self.in_channels) +
                len(self.filters.shape[2:])*(1,)).expand(self.filters.shape)
            self.filters.div_(norms)

    def reconstructions(self, activations):
        return self.deconvo(activations, self.filters, padding=self.padding,
                            stride=self.stride)

    def loss(self, images, activations):
        reconstructions = self.reconstructions(activations)
        loss = 0.5 * (1/images.shape[0]) * torch.sum(
            torch.pow(images - reconstructions, 2))
        loss += self.lam * torch.mean(torch.sum(torch.abs(
            activations.reshape(activations.shape[0], -1)), dim=1))
        return loss

    def u_grad(self, u, images):
        acts = self.threshold(u)
        recon = self.reconstructions(acts)
        e = images - recon
        du = -u
        du += self.convo(e, self.filters, padding=self.padding,
                         stride=self.stride)
        du += acts
        return du
    
    def get_output_shape(self, images):
        output_shape = []
        if self.conv_dim >= 1:
            output_shape.append(math.floor(((images.shape[2] + 2 *
                                           self.padding[0] -
                                           (self.kernel_size[0] - 1) - 1) /
                                          self.stride[0]) + 1))
        if self.conv_dim >= 2:
            output_shape.append(math.floor(((images.shape[3] + 2 *
                                           self.padding[1] -
                                           (self.kernel_size[1] - 1) - 1) /
                                          self.stride[1]) + 1))
        if self.conv_dim >= 3:
            output_shape.append(math.floor(((images.shape[4] + 2 *
                                           self.padding[2] -
                                           (self.kernel_size[2] - 1) - 1) /
                                          self.stride[2]) + 1))
            
        return output_shape
        

    def activations(self, images, u_init):
        with torch.no_grad():
            output_shape = self.get_output_shape(images)
            # print('input shape', images.shape)
            # print('output shape', output_shape)

#             u = torch.zeros([images.shape[0], self.out_channels] +
#                     output_shape, device=self.filters.device)
#             u = torch.full([images.shape[0], self.out_channels] +
#                     output_shape, fill_value=self.lam, device=self.filters.device)
            u = u_init.detach().clone().to(self.filters.device)
#             for i in range(self.max_activation_iter):
#                 du = self.u_grad(u, images)
# #                 print(torch.sum(du))
#                 # print("grad_norm={}, iter={}".format(torch.norm(du), i))
#                 u += self.activation_lr * du
#                 if torch.norm(du) < 0.01:
#                     break
            b1 = 0.9
            b2 = 0.999
            eps = 1e-8
            m = torch.zeros_like(u)
            v = torch.zeros_like(u)
            for i in range(self.max_activation_iter):
                g = self.u_grad(u, images)
                m = b1 * m + (1-b1) * g
                v = b2 * v + (1-b2) * g**2
                mh = m / (1 - b1**(i+1))
                vh = v / (1 - b2**(i+1))
                u += self.activation_lr * mh / (torch.sqrt(vh) + eps)

        return self.threshold(u), u

    def forward(self, images, u_init):
        return self.activations(images, u_init)
    
    
    def import_opencv_dir(self, in_dir):
        i = 0
        for f in sorted(os.listdir(in_dir), key=lambda x: str(x)):
            if not f.endswith('.mat'):
                continue
            mat = scipy.io.loadmat(os.path.join(in_dir, f))

            dic = torch.from_numpy((mat['weight_vals'].astype(np.float32)))

            dic = dic.permute(2,1,0).unsqueeze(1)

            dic = dic.float() /1

            if self.filters.data[:,:,i,:,:].size() != dic.size():
                raise Exception('Input dictionary size is: ' + str(dic.size()) + ' while model filter size is: ' + str(self.filters.data[:,:,i,:,:].size()))

            self.filters.data[:,:,i,:,:] = dic

            i += 1

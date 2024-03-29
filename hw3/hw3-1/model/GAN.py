import torch.nn as nn
import torch.nn.functional as F
import numpy as np

img_shape = (3, 64, 64)
latent_dim = 100
 
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()

#         def block(in_feat, out_feat, normalize=True):
#             layers = [nn.Linear(in_feat, out_feat)]
#             if normalize:
#                 layers.append(nn.BatchNorm1d(out_feat, 0.8))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers

#         self.model = nn.Sequential(
#             *block(latent_dim, 128, normalize=False),
#             *block(128, 256),
#             *block(256, 512),
#             *block(512, 1024),
#             nn.Linear(1024, int(np.prod(img_shape))),
#             nn.Tanh()
#         )

#     def forward(self, z):
#         flat = z.view(z.shape[0], -1)
#         img = self.model(flat)
#         img = img.view(img.size(0), *img_shape)
#         return img

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        relu = nn.LeakyReLU(0.2, inplace=True) # nn.ReLU(inplace=True)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(relu)
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        flat = z.view(z.shape[0], -1)
        img = self.model(flat)
        img = img.view(img.size(0), *img_shape)
        return img
        
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        relu = nn.LeakyReLU(0.2, inplace=True) # nn.ReLU(inplace=True)

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            relu,
            nn.Linear(512, 256),
            relu,
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

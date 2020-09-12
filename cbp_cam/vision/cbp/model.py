from __future__ import division
import os

import torch
import torchvision
import torch.nn.functional as F

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dim = 8192
par_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sketch_matrix01 = torch.load(par_dir + '/data/sketch_matrix1.pth', map_location='cpu')
sketch_matrix02 = torch.load(par_dir + '/data/sketch_matrix2.pth', map_location='cpu')

class BCNN(torch.nn.Module):
    """B-CNN for CUB200.

    The B-CNN model is illustrated as follows.
    conv1^2 (64) -> pool1 -> conv2^2 (128) -> pool2 -> conv3^3 (256) -> pool3
    -> conv4^3 (512) -> pool4 -> conv5^3 (512) -> bilinear pooling
    -> sqrt-normalize -> L2-normalize -> fc (200).
    The network accepts a 3*448*448 input, and the pool5 activation has shape
    512*28*28 since we down-sample 5 times.

    Attributes:
        features, torch.nn.Module: Convolution and pooling layers.
        fc, torch.nn.Module: 200.
    """

    def __init__(self):
        """Declare all needed layers."""
        torch.nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.features = torchvision.models.vgg16(pretrained=False).features
        self.features = torch.nn.Sequential(*list(self.features.children())
        [:-1])  # Remove pool5.
        # Linear classifier.
        self.fc = torch.nn.Linear(output_dim, 200)

    def forward(self, X):
        """Forward pass of the network.

        Args:
            X, torch.autograd.Variable of shape N*3*448*448.

        Returns:
            Score, torch.autograd.Variable of shape N*200.
        """

        N = X.size()[0]
        assert X.size() == (N, 3, 448, 448)
        X = self.features(X)
        assert X.size() == (N, 512, 28, 28)
        sketch_matrix1 = sketch_matrix01
        sketch_matrix2 = sketch_matrix02
        fft1 = torch.rfft(X.permute(0, 2, 3, 1).matmul(sketch_matrix1), 1)
        fft2 = torch.rfft(X.permute(0, 2, 3, 1).matmul(sketch_matrix2), 1)
        fft_product = torch.stack([fft1[..., 0] * fft2[..., 0] - fft1[..., 1] * fft2[..., 1],
                                   fft1[..., 0] * fft2[..., 1] + fft1[..., 1] * fft2[..., 0]], dim=-1)
        cbp = torch.irfft(fft_product, 1, signal_sizes=(output_dim,)) * output_dim
        X = cbp.sum(dim=1).sum(dim=1)
        X = torch.sqrt(F.relu(X)) - torch.sqrt(F.relu(-X))
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        assert X.size() == (N, 200)
        return X


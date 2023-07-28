import torch
import torch.nn as nn

from .ghiasi import Ghiasi
from .stylePredictor import StylePredictor
import numpy as np
import sys
from os.path import join, dirname

class StyleAugmentor(nn.Module):
    def __init__(self):
        super(StyleAugmentor,self).__init__()

        # create transformer and style predictor networks:
        self.ghiasi = Ghiasi()
        # self.stylePredictor = StylePredictor()
        self.ghiasi.cuda()
        # self.stylePredictor.cuda()

        # load checkpoints:
        checkpoint_ghiasi = torch.load(join(dirname(__file__),'checkpoints/checkpoint_transformer.pth'))
        # checkpoint_stylepredictor = torch.load(join(dirname(__file__),'checkpoints/checkpoint_stylepredictor.pth'))
        checkpoint_embeddings = torch.load(join(dirname(__file__),'checkpoints/checkpoint_embeddings.pth'))
        
        # load weights for ghiasi and stylePredictor, and mean / covariance for the embedding distribution:
        self.ghiasi.load_state_dict(checkpoint_ghiasi['state_dict_ghiasi'],strict=False)
        # self.stylePredictor.load_state_dict(checkpoint_stylepredictor['state_dict_stylepredictor'],strict=False)

        # load mean imagenet embedding:
        self.imagenet_embedding = checkpoint_embeddings['imagenet_embedding_mean'] # mean style embedding for ImageNet
        self.imagenet_embedding = self.imagenet_embedding.cuda()

        # get mean and covariance of PBN style embeddings:
        self.mean = checkpoint_embeddings['pbn_embedding_mean']
        self.mean = self.mean.cuda() # 1 x 100
        self.cov = checkpoint_embeddings['pbn_embedding_covariance']
        
        # compute SVD of covariance matrix:
        u, s, vh = np.linalg.svd(self.cov.numpy())
        
        self.A = np.matmul(u,np.diag(s**0.5))
        self.A = torch.tensor(self.A).float().cuda() # 100 x 100
        # self.cov = cov(Ax), x ~ N(0,1)
    
    def sample_embedding(self, n):
        # n: number of embeddings to sample
        # returns n x 100 embedding tensor
        embedding = torch.randn(n,100).cuda() # n x 100
        embedding = torch.mm(embedding,self.A.transpose(1,0)) + self.mean # n x 100
        return embedding

    def forward(self, x, alpha=1.0, downsamples=0, embedding=None):
        # augments a batch of images with style randomization
        # x: B x C x H x W image tensor
        # alpha: float in [0,1], controls interpolation between random style and original style
        # downsamples: int, number of times to downsample by factor of 2 before applying style transfer
        # embedding: B x 100 tensor, or None. Use this embedding if provided.
        # style embedding for when alpha=0:

        if downsamples:
            assert(x.size(2) % 2**downsamples == 0)
            assert(x.size(3) % 2**downsamples == 0)
            for i in range(downsamples):
                x = nn.functional.avg_pool2d(x,2)

        if embedding is None:
            # sample a random embedding
            embedding = self.sample_embedding(x.size(0))
        # interpolate style embeddings:
        embedding = alpha*embedding
        
        restyled = self.ghiasi(x,embedding)

        if downsamples:
            restyled = nn.functional.upsample(restyled,scale_factor=2**downsamples,mode='bilinear')
        
        return restyled.detach() # detach prevents the user from accidentally backpropagating errors into stylePredictor or ghiasi while training a downstream model

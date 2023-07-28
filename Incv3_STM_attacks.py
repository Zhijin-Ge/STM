"""Implementation of sample attack."""
import os
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
from attack_methods import DI, gkern, Admix
from torchvision import transforms as T
from torchvision.transforms import ToTensor, ToPILImage, transforms
from tqdm import tqdm
import numpy as np
from PIL import Image
from Normalize import Normalize
from loader import ImageNet
from torch.utils.data import DataLoader
import argparse
import pretrainedmodels
from os.path import join, dirname
from styleaug import StyleAugmentor


parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='./dataset/images.csv', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='./dataset/images', help='Input directory with images.')
parser.add_argument('--output_dir', type=str, default='./outputs/', help='Output directory with adversarial images.')
parser.add_argument('--mean', type=float, default=np.array([0.5, 0.5, 0.5]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.5, 0.5, 0.5]), help='std.')
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter_set", type=int, default=10, help="Number of iterations.")
parser.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=299, help="Height of each input images.")
parser.add_argument("--batch_size", type=int, default=10, help="How many images process at one time.")
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")
parser.add_argument("--N", type=int, default=20, help="The number of Spectrum Transformations (Sampling Number)")
parser.add_argument("--portion", type=float, default=0.2, help="protion for the mixed image")
parser.add_argument("--gamma", type=float, default=0.5, help="protion for the mixed original image")
parser.add_argument("--beta", type=float, default=2.0, help="weighted")
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

transforms = T.Compose(
    [T.Resize(299), T.ToTensor()]
)

def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def save_image(images,names,output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    for i,name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))
        img.save(output_dir + name)

T_kernel = gkern(7, 3)

def STM(images, gt, model, min, max): 
    """
    The attack algorithm of our proposed Style Transfer Method
    :param images: the input images
    :param gt: ground-truth
    :param model: substitute model
    :param mix: the mix the clip operation 
    :param max: the max the clip operation
    :return: the adversarial images
    """
    Resize = T.Resize(size=(299, 299))
    momentum = opt.momentum
    num_iter = opt.num_iter_set
    eps = opt.max_epsilon / 255.0
    alpha = eps / num_iter
    x = images.clone()
    grad = 0
    N = opt.N
    beta = opt.beta
    gamma = opt.gamma
    # create style augmentor:
    augmentor = StyleAugmentor()

    for i in range(num_iter):
        noise = 0
        for n in range(N):
            # randomize style:
            x_aug = augmentor(x)
            x_new = gamma*x + (1-gamma)*Resize(x_aug.detach()).clone() + torch.randn_like(x).uniform_(-eps*beta, eps*beta)
            x_new = V(x_new, requires_grad = True)

            output_v3 = model(x_new)
            loss = F.cross_entropy(output_v3, gt)
            noise += torch.autograd.grad(loss, x_new,
                                        retain_graph=False, create_graph=False)[0]
        noise = noise / N

        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise

        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, min, max)
    return x.detach()


def MIFGSM(images, gt, model, min, max):
    """
    The attack algorithm of MI-FGSM
    :param images: the input images
    :param gt: ground-truth
    :param model: substitute model
    :param mix: the mix the clip operation 
    :param max: the max the clip operation
    :return: the adversarial images
    """
    grad = 0
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter_set
    alpha = eps / 10.0
    momentum = opt.momentum
    x = images.clone().detach().cuda()

    for i in range(num_iter):
        x = V(x, requires_grad = True)

        output_v3 = model(x)
        loss = F.cross_entropy(output_v3, gt)

        noise = torch.autograd.grad(loss, x,
                                    retain_graph=False, create_graph=False)[0]
        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise
        
        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, min, max)
    return x.detach()


def NIFGSM(images, gt, model, min, max):
    """
    The attack algorithm of NI-FGSM
    :param images: the input images
    :param gt: ground-truth
    :param model: substitute model
    :param mix: the mix the clip operation 
    :param max: the max the clip operation
    :return: the adversarial images
    """
    grad = 0
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter_set
    alpha = eps / num_iter
    momentum = opt.momentum
    x = images.clone().detach().cuda()

    for i in range(num_iter):
        x = V(x, requires_grad = True)
        nes_x = x + momentum * alpha * grad
        output_v3 = model(nes_x)
        loss = F.cross_entropy(output_v3, gt)

        noise = torch.autograd.grad(loss, x,
                                    retain_graph=False, create_graph=False)[0]
        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise
        
        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, min, max)
    return x.detach()


def DIM(images, gt, model, min, max):
    """
    The attack algorithm of DIM
    :param images: the input images
    :param gt: ground-truth
    :param model: substitute model
    :param mix: the mix the clip operation 
    :param max: the max the clip operation
    :return: the adversarial images
    """
    grad = 0
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter_set
    alpha = eps / num_iter
    momentum = opt.momentum
    x = images.clone().detach().cuda()

    for i in range(num_iter):
        x = V(x, requires_grad = True)
        output_v3 = model(DI(x))
        
        loss = F.cross_entropy(output_v3, gt)

        noise = torch.autograd.grad(loss, x,
                                    retain_graph=False, create_graph=False)[0]
        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise
        
        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, min, max)
    return x.detach()


def TIM(images, gt, model, min, max):
    """
    The attack algorithm of TIM
    :param images: the input images
    :param gt: ground-truth
    :param model: substitute model
    :param mix: the mix the clip operation 
    :param max: the max the clip operation
    :return: the adversarial images
    """
    grad = 0
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter_set
    alpha = eps / num_iter
    momentum = opt.momentum
    x = images.clone().detach().cuda()
    T_kernel = gkern(7, 3)
    for i in range(num_iter):
        x = V(x, requires_grad = True)

        output_v3 = model(x)
        loss = F.cross_entropy(output_v3, gt)

        noise = torch.autograd.grad(loss, x,
                                    retain_graph=False, create_graph=False)[0]
        noise = F.conv2d(noise, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)
        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise
        
        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, min, max)
    return x.detach()

def SIM(images, gt, model, min, max):
    """
    The attack algorithm of SIM
    :param images: the input images
    :param gt: ground-truth
    :param model: substitute model
    :param mix: the mix the clip operation 
    :param max: the max the clip operation
    :return: the adversarial images
    """
    grad = 0
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter_set
    alpha = eps / num_iter
    momentum = opt.momentum
    x = images.clone().detach().cuda()

    for i in range(num_iter):
        x = V(x, requires_grad = True)
        noise = torch.zeros_like(x).detach().cuda()
        for i in torch.arange(5):
            nes_x = x / torch.pow(2, i)
            output_v3 = model(nes_x)
            loss = F.cross_entropy(output_v3, gt)
            noise += torch.autograd.grad(loss, x,
                                        retain_graph=False, create_graph=False)[0]
        noise = noise / 5
        
        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise
        
        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, min, max)
    return x.detach()


def Admix_MI(images, gt, model, min, max):
    """
    The attack algorithm of Admix
    :param images: the input images
    :param gt: ground-truth
    :param model: substitute model
    :param mix: the mix the clip operation 
    :param max: the max the clip operation
    :return: the adversarial images
    """
    grad = 0
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter_set
    alpha = eps / num_iter
    momentum = opt.momentum
    portion = opt.portion
    size = opt.size
    x = images.clone().detach().cuda()
    g_t = torch.cat([torch.cat([(gt) for _ in range(size)]) for _ in range(5)]) 
    for i in range(num_iter):
        x = V(x, requires_grad = True)
        admix = Admix(x, size, portion)
        x_batch = torch.cat([admix, admix/2, admix/4, admix/8, admix/16], axis=0)
        output_v3 = model(x_batch)
        loss = F.cross_entropy(output_v3, g_t)

        noise_total = torch.autograd.grad(loss, x_batch,
                                    retain_graph=False, create_graph=False)[0]
        noise1, noise2, noise3, noise4, noise5 = torch.split(noise_total, x.shape[0]*size, dim=0)
        avg1_noise = 1/5*(noise1+noise2/2+noise3/4+noise4/8+noise5/16)
        avg1, avg2, avg3 = torch.split(avg1_noise, x.shape[0], dim=0)
        noise = 1/3*(avg1+avg2+avg3)
        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise
        
        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, min, max)
    return x.detach()


def main():

    model = torch.nn.Sequential(Normalize(opt.mean, opt.std),
                                pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().cuda())
    
    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    for images, images_ID,  gt_cpu in tqdm(data_loader):
        gt = gt_cpu.cuda()
        images = images.cuda()
        images_min = clip_by_tensor(images - opt.max_epsilon / 255.0, 0.0, 1.0)
        images_max = clip_by_tensor(images + opt.max_epsilon / 255.0, 0.0, 1.0)

        adv_img5 = STM(images, gt, model, images_min, images_max)
        adv_img_np5 = adv_img5.cpu().numpy()
        adv_img_np5 = np.transpose(adv_img_np5, (0, 2, 3, 1)) * 255
        save_image(adv_img_np5, images_ID, f'./incv3_stm_outputs/')


if __name__ == '__main__':
    main()
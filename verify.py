"""Implementation of evaluate attack result."""
import os
import torch
from torch.autograd import Variable as V
from torch import nn
from torchvision import transforms as T
from Normalize import Normalize, TfNormalize
from loader import ImageNet
from torch.utils.data import DataLoader
import pretrainedmodels

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

batch_size = 10

input_csv = './dataset/images.csv'
input_dir = './dataset/images'
adv_dir = './incv3_stm_outputs'

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def get_model(net_name, model_dir):
    """Load converted model"""
    model_path = os.path.join(model_dir, net_name + '.npy')

    if net_name == 'inception_v3':
        model = torch.nn.Sequential(Normalize(mean, std),
                                pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'inception_v4':
        model = torch.nn.Sequential(Normalize(mean, std),
                                pretrainedmodels.inceptionv4(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'resnet_v2_50':
        model = torch.nn.Sequential(Normalize(mean, std),
                                pretrainedmodels.resnet50(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'resnet_v2_101':
        model = torch.nn.Sequential(Normalize(mean, std),
                                pretrainedmodels.resnet101(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'resnet_v2_152':
        model = torch.nn.Sequential(Normalize(mean, std),
                                pretrainedmodels.resnet152(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'inc_res_v2':
        model = torch.nn.Sequential(Normalize(mean, std),
                                pretrainedmodels.inceptionresnetv2(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'tf_adv_inception_v3':
        from torch_nets import tf_adv_inception_v3
        net = tf_adv_inception_v3
        model = nn.Sequential( 
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval().cuda(),)
    elif net_name == 'tf_ens3_adv_inc_v3':
        from torch_nets import tf_ens3_adv_inc_v3
        net = tf_ens3_adv_inc_v3
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval().cuda(),)
    elif net_name == 'tf_ens4_adv_inc_v3':
        from torch_nets import tf_ens4_adv_inc_v3
        net = tf_ens4_adv_inc_v3
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval().cuda(),)
    elif net_name == 'tf_ens_adv_inc_res_v2':
        from torch_nets import tf_ens_adv_inc_res_v2
        net = tf_ens_adv_inc_res_v2
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval().cuda(),)
    else:
        print('Wrong model name!')

    return model

def verify(model_name, path):
    img_size = 299
    model = get_model(model_name, path)

    X = ImageNet(adv_dir, input_csv, T.Compose([T.ToTensor(), T.Resize(img_size)]))
    data_loader = DataLoader(X, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    sum = 0
    for images, _, gt_cpu in data_loader:
        gt = gt_cpu.cuda()
        images = images.cuda()
        with torch.no_grad():
            sum += (model(images).argmax(1) != (gt)).detach().sum().cpu()

    print(model_name + '  acu = {:.2%}'.format(sum / 1000.0))

def main():
    model_names = ['inception_v3', 'inception_v4', 'inc_res_v2', 'resnet_v2_50', 'resnet_v2_101', 'resnet_v2_152']
    models_path = './models/'
    for model_name in model_names:
        verify(model_name, models_path)
        print("===================================================")

if __name__ == '__main__':
    print(adv_dir)
    main()
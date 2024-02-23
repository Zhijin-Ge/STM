# STM
This repository contains the PyTorch code for the paper:

[Improving the Transferability of Adversarial Examples with Arbitrary Style Transfer](https://arxiv.org/abs/2308.10601).

which has been accepted by ACM MM 2023.
> We also include the code in the framework [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack).
## Requirements
* python == 3.7.11
* pytorch == 1.8.0
* torchvision == 0.8.0
* numpy == 1.21.2
* pandas == 1.3.5
* opencv-python == 4.5.4.60
* scipy == 1.7.3
* pillow == 8.4.0
* pretrainedmodels == 0.7.4
* tqdm == 4.62.3
* imageio == 2.6.1


## Qucik Start
### Prepare the data and models.
1. We have prepared the ImageNet-compatible dataset in this program and put the data in **'./dataset/'**.

2. The normally trained models (i.e., Inc-v3, Inc-v4, IncRes-v2, Res-50, Res-101, Res-100) are from "pretrainedmodels", if you use it for the first time, it will download the weight of the model automatically, just wait for it to finish. 

3. The adversarially trained models (i.e, ens3_adv_inc_v3, ens4_adv_inc_v3, ens_adv_inc_res_v2) are from [SSA](https://github.com/yuyang-long/SSA) or [tf_to_torch_model](https://github.com/ylhz/tf_to_pytorch_model). For more detailed information on how to use them, visit these two repositories.

### STM Attack Method
The traditional baseline attacks and our proposed STM attack methods are in the file __"Incv3_STM_Attacks.py"__.
All the provided codes generate adversarial examples on Inception_v3 model. If you want to attack other models, replace the model in **main()** function.

### Runing attack
1. You can run our proposed attack as follows. 
```
python Incv3_STM_Attacks.py
```
We also provide the implementations of other baseline attack methods in our code, just change them to the corresponding attack methods in the **main()** function.

2. The generated adversarial examples would be stored in the directory **./incv3_xx_xx_outputs**. Then run the file **verify.py** to evaluate the success rate of each model used in the paper:
```
python verify.py
```
## Acknowledgments
The codes mainly references: [SSA](https://github.com/yuyang-long/SSA) and [styleAug](https://github.com/philipjackson/style-augmentation)

## Citation
If you use this code for your research, please cite our paper.
```
@inproceedings{ge2023improving,
     title={{Improving the Transferability of Adversarial Examples with Arbitrary Style Transfer}},
     author={Zhijin Ge and Fanhua Shang and Hongying Liu and Yuanyuan Liu and Liang Wan and Wei Feng and Xiaosen Wang},
     booktitle={Proceedings of the ACM International Conference on Multimedia},
     year={2023}
}
```

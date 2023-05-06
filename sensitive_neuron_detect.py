import numpy as np
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, SVHN

from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F
from torchvision.transforms import ToTensor

from resnet20 import ResNet20
from VGG16 import VGG16
from LeNet import LeNet5, LeNet1
from benign_perturbations import benign_aug



if __name__ == '__main__':
    
    test_dataset = CIFAR10(root='../data', train=False, transform=ToTensor(), download=True)
    # test_dataset = SVHN(root='../data', split='test', transform=ToTensor(), download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)

    
    dataset = 'cifar10'
    model = 'resnet20'
    testmodel = torch.load('./pretrained/'+dataset+'_'+model+'_0.861.pt')

    # testmodel = ResNet20().cuda()
    # testmodel.load_state_dict(torch.load('./checkpoint/cifar10_epoch_200.pth'), strict=True)

    benign_params = {
                    'shift_x':(0.05,0.15),
                    'shift_y':(0.05,0.15),
                    'rotate':(5,25),
                    'scale':(0.8,1.2),
                    'shear':(15,30),
                    'contrast':(0.5,1.5),
                    'brightness':(0.5,1.5),
                    'blur_mode':'easy'}
    benign_aug = benign_aug(params=benign_params, model = testmodel, seed = 0)


    all_correct_num = 0
    all_sample_num = 0
    testmodel.eval()



    acc_array = []
    neuron_diff_list = []
    
    with tqdm(test_loader) as loader:
        for idx, (test_x, test_label) in enumerate(loader):
            test_x = test_x.cuda()

            predict_y_adv, clean_features = testmodel(test_x)
            test_x_adv = benign_aug(test_x)
            predict_y_adv, adv_features = testmodel(test_x_adv)

            neuron_diff = F.mse_loss(adv_features, clean_features, reduction='none').flatten(start_dim=1) 
            neuron_num = neuron_diff.shape[1]
            neuron_diff /= clean_features.flatten(start_dim=1).max(dim=1)[0].unsqueeze(1).repeat(1,neuron_num)
            

            neuron_diff_list.append(neuron_diff.mean(dim=0).detach().cpu().numpy())
            selected_neurons_num = neuron_num // 10
            batch_neuron_idx += torch.argsort(neuron_diff)[:,-selected_neurons_num:].flatten().cpu()
 
            predict_y_adv = np.argmax(predict_y_adv.cpu().detach(), axis=-1)
            current_correct_num = predict_y_adv == test_label
            all_correct_num = current_correct_num.sum()
            all_sample_num = current_correct_num.shape[0]
            acc = all_correct_num / all_sample_num

            acc_array.append(acc)

    sensitive_neuron_idx = np.argsort(np.mean(neuron_diff_list, axis=0))[-selected_neurons_num:]
    np.save('./'+dataset+'-'+model+'-sensNeurons.npy', sensitive_neuron_idx)


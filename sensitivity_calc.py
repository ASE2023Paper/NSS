import numpy as np
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, SVHN
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F
from torchvision.transforms import ToTensor
# from feature_wise_pgd import PGD
import random
from PGD import PGD, FGSM
# from mcmc import MCMC
from VGG16 import VGG16
from resnet20 import ResNet20
from LeNet import LeNet5, LeNet1
from benign_perturbations import benign_aug
from feature_map_plot import feature_map_plot

import torchattacks

if __name__ == '__main__':
    
    batch_size = 10

    test_dataset = CIFAR10(root='../data', train=False, transform=ToTensor(), download=True)
    # test_dataset = SVHN(root='../data', split='test', transform=ToTensor(), download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last = True)

    

    dataset = 'cifar10'
    model = 'resnet20'
    testmodel = torch.load('./pretrained/'+dataset+'_'+model+'_0.861.pt').cuda()

    neuron_idx = np.load('./'+ dataset + '-' + model +'-sensNeurons.npy')
    sampled_neurons_num = int(neuron_idx.shape[0])
    neuron_idx = neuron_idx[:sampled_neurons_num]

    ''' load retrained model state dict'''
    # testmodel = VGG16().cuda()
    # testmodel.load_state_dict(torch.load('./retrained/'+dataset+'_'+model+'_p20.pt'), strict=True)


    benign_params_1 = {
                    'shift_x':(0.0,0.05),
                    'shift_y':(0.0,0.05),
                    'rotate':(0,5),
                    'scale':(0.95,1.05),
                    'shear':(0,5),
                    'contrast':(0.9,1.1),
                    'brightness':(0.9,1.1),
                    'blur_mode':'easy'}
    benign_params_2 = {
                    'shift_x':(0.05,0.10),
                    'shift_y':(0.05,0.10),
                    'rotate':(5,10),
                    'scale':(0.9,1.1),
                    'shear':(5,10),
                    'contrast':(0.7,1.3),
                    'brightness':(0.7,1.3),
                    'blur_mode':'easy'}
    benign_params_3 = {
                    'shift_x':(0.10,0.15),
                    'shift_y':(0.10,0.15),
                    'rotate':(10,15),
                    'scale':(0.8,1.2),
                    'shear':(10,15),
                    'contrast':(0.5,1.5),
                    'brightness':(0.5,1.5),
                    'blur_mode':'easy'}
    benign_params_4 = {
                    'shift_x':(0.15,0.20),
                    'shift_y':(0.15,0.20),
                    'rotate':(15,20),
                    'scale':(0.7,1.3),
                    'shear':(15,20),
                    'contrast':(0.4,1.6),
                    'brightness':(0.4,1.6),
                    'blur_mode':'easy'}
    benign_params_5 = {
                    'shift_x':(0.20,0.25),
                    'shift_y':(0.20,0.25),
                    'rotate':(20,25),
                    'scale':(0.6,1.4),
                    'shear':(20,25),
                    'contrast':(0.3,1.7),
                    'brightness':(0.3,1.7),
                    'blur_mode':'easy'}
    mutation_params_diff_level = [benign_params_1, benign_params_2, benign_params_3, benign_params_4, benign_params_5]
    
    Sensitivity_levels = []
    Sensitivity_levels_random = []
    np.random.seed(0)
    
    for benign_params in mutation_params_diff_level:
        mutation = benign_aug(params=benign_params, model = testmodel, seed = 0)
        all_correct_num = 0
        all_sample_num = 0
        testmodel.eval()


        correct_imgs = []
        correct_labels = []

        acc_array = []
        per_feature_acc = []

        batch_neuron_idx = []
        neuron_dict = {}
        # print(random_idx)
        # random_idx = [11474, 3489, 12733, 13612, 16289]
        # print(random_idx)
        total_diff = 0
        total_diff_random = 0
        # most_sensitive_samples = []
        # most_sensitive_samples_label = []
        neuron_diff_perSample = []
        neuron_diff_perSample_random = []
        all_adv_samples = []
        all_labels = []
        total_neurons = 0
        avg_feature_diff = []

        total_activated_num_list = []

        clean_activation_mean = []

        with tqdm(test_loader) as loader:
            for idx, (test_x, test_label) in enumerate(loader):
                test_x = test_x.cuda()

                predict_y_adv, clean_features = testmodel(test_x)

                test_x_adv = mutation(test_x)
                clean_activation_mean.append(clean_features.flatten(start_dim=1).mean(dim=0).detach().cpu().numpy())

                predict_y_adv, adv_features = testmodel(test_x_adv)

  
                neuron_diff = F.mse_loss(adv_features, clean_features, reduction='none').flatten(start_dim=1)
                neuron_num = neuron_diff.shape[1]
                neuron_diff /= clean_features.flatten(start_dim=1).max(dim=1)[0].unsqueeze(1).repeat(1,neuron_num)
   
                random_idx = np.random.choice(np.linspace(0, neuron_num-1, neuron_num, dtype=np.int32), sampled_neurons_num)
                neuron_diff_perSample.append(neuron_diff[:, neuron_idx].mean(dim=1).detach().cpu().numpy())
                neuron_diff_perSample_random.append(neuron_diff[:, random_idx].mean(dim=1).detach().cpu().numpy())

                selected_neurons_num = 1000 if neuron_num > 1000 else neuron_num
                batch_neuron_idx += torch.argsort(neuron_diff)[:,-selected_neurons_num:].flatten().cpu()
                # print(torch.argsort(max_diff_neurons)[-5:])


                # print(test_label)
                predict_y_adv = np.argmax(predict_y_adv.cpu().detach(), axis=-1)
                current_correct_num = predict_y_adv == test_label

                # print(current_correct_num.shape)
                all_correct_num = current_correct_num.sum()
                all_sample_num = current_correct_num.shape[0]
                acc = all_correct_num / all_sample_num
                # print(all_correct_num)
                acc_array.append(acc)


        Sensitivity_levels.append(np.mean(neuron_diff_perSample) / sampled_neurons_num)
        Sensitivity_levels_random.append(np.mean(neuron_diff_perSample_random) / sampled_neurons_num)

    print(Sensitivity_levels)
    print(Sensitivity_levels_random)
 

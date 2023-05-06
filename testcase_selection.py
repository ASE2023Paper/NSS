import numpy as np
import os
import torch
from tqdm import tqdm
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, SVHN
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import random
from resnet20 import ResNet20
from LeNet import LeNet5, LeNet1
from VGG16 import VGG16
from benign_perturbations import benign_aug

import time


if __name__ == '__main__':

    test_dataset = CIFAR10(root='../data', train=False, transform=ToTensor(), download=True)
    # test_dataset = SVHN(root='../data', split='test', transform=ToTensor(), download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


    dataset = 'cifar10'
    model = 'resnet20'
    testmodel = torch.load('./pretrained/'+dataset+'_'+model+'_0.861.pt')

    benign_params = {
                    'shift_x':(0.05,0.15),
                    'shift_y':(0.05,0.15),
                    'rotate':(5,25),
                    'scale':(0.8,1.2),
                    'shear':(15,30),
                    'contrast':(0.5,1.5),
                    'brightness':(0.5,1.5),
                    'blur_mode':'easy'}
    benign_aug = benign_aug(params=benign_params, model = testmodel, seed = 123)


    all_correct_num = 0
    all_sample_num = 0
    testmodel.eval()
    

    neuron_idx = np.load('./'+ dataset + '-' + model +'-sensNeurons.npy')
    sampled_neurons_num = int(neuron_idx.shape[0])
    neuron_idx = neuron_idx[-sampled_neurons_num:]

    neuron_diff_perSample = []
    all_adv_samples = []
    all_labels = []
    acc_array = []

    start = time.time()
    with tqdm(test_loader) as loader:
        for idx, (test_x, test_label) in enumerate(loader):

            test_x = test_x.cuda()
            test_x_adv = benign_aug(test_x)
            predict_y_adv, adv_features = testmodel(test_x_adv)


            neuron_diff = F.mse_loss(adv_features, clean_features, reduction='none').flatten(start_dim=1)
            neuron_diff /= clean_features.flatten(start_dim=1).max(dim=1)[0].unsqueeze(1).repeat(1,neuron_num)
            neuron_diff_perSample.append(neuron_diff[:, neuron_idx].mean(axis=1))


            all_adv_samples.append(test_x_adv.cpu().numpy())
            all_labels.append(test_label.cpu().numpy())

            predict_y_adv = np.argmax(predict_y_adv.cpu().detach(), axis=-1)
            current_correct_num = predict_y_adv == test_label

            all_correct_num = current_correct_num.sum()
            all_sample_num = current_correct_num.shape[0]
            acc = all_correct_num / all_sample_num

            acc_array.append(acc)



    sensitive_sample_idx = np.argsort(np.array(neuron_diff_perSample).flatten())[-2000:]

    all_adv_samples = np.array(all_adv_samples)
    all_labels = np.array(all_labels)

    b, bs, c, w, h = all_adv_samples.shape
    most_sensitive_samples = all_adv_samples.reshape(b*bs, c, w, h)[sensitive_sample_idx]
    most_sensitive_samples_label = all_labels.reshape(b * bs)[sensitive_sample_idx]
    end = time.time()

    print('time overhead: ', end-start)

    ''' Save dataset selected by NSS'''
    np.save('./neuron_sensitive_samples_' + dataset + '_' + model + '_images.npy', most_sensitive_samples)
    np.save('./neuron_sensitive_samples_' + dataset + '_' + model + '_labels.npy', most_sensitive_samples_label)

    print('Attack Success Rate: ', 1 - np.array(acc_array).mean())
    print('Acc: ', np.array(acc_array).mean())
    print('Total Correct Num: ', np.array(acc_array).sum() * batch_size)


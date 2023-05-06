import numpy as np
import argparse
import torch
from tqdm import tqdm
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, SVHN

from torch.utils.data import DataLoader,TensorDataset
from torchvision.transforms import ToTensor

from resnet20 import ResNet20
from benign_perturbations import benign_aug
from adversarial_perturbations import adv_aug



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='testcases generation')
    parser.add_argument('--save_path', type=str, default='../data/', help='saved data path')
    parser.add_argument('--data_root', type=str, default='../data/', help='dataset path')
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--benign', help='perturbation mode:adversarial', action='store_true')
    group.add_argument('--adv', help='perturbation mode:benign', action='store_true')
    args = parser.parse_args()

    # test_x_numpy = np.load('./correct_imgs_cifar10_resnet20.npy')
    # test_y_numpy = np.load('./correct_labels_cifar10_resnet20.npy')

    # x_test=torch.from_numpy(test_x_numpy)
    # y_test=torch.from_numpy(test_y_numpy)

    # print(y_test)
    # x_test = torch.flip(x_test, dims=[0])
    # y_test = torch.flip(y_test, dims=[0])
    # test_dataset=TensorDataset(x_test,y_test)
    test_dataset = CIFAR10(root=args.data_root, train=False, transform=ToTensor(), download=True)
    # test_dataset = SVHN(root=args.data_root, split='test', transform=ToTensor(), download=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # testmodel = torch.load('./pretrained/mnist_lenet5_0.918.pt').cuda()
    testmodel = ResNet20(num_classes=10).cuda()
    testmodel.load_state_dict(torch.load('./checkpoint/cifar10_epoch_200.pth'), strict=True)

    if args.benign:
        benign_params = {'shift_x':(0.05,0.15),
                        'shift_y':(0.05,0.15),
                        'rotate':(5,25),
                        'scale':(0.7,1.5),
                        'shear':(15,30),
                        'contrast':(0.5,1.5),
                        'brightness':(0.5,1.5),
                        'blur_mode':'easy'}
        benign_aug = benign_aug(params=benign_params, model = testmodel, seed = 0)
    elif args.adv:
        adv_params = {'eps':16/255,
                    'alpha':8/255,
                    'steps':1,
                    'random_start':True}
        adv_aug = adv_aug(params=adv_params, model = testmodel)
    
    


    testmodel.eval()

    label = []
    npy_adv_images = []
    diff_array = []
    feature_idx_array = []
    with tqdm(test_loader) as loader:
        for idx, (test_x, test_label) in enumerate(loader):
            # if idx==1000:
            #     break
            
            test_x = test_x.cuda()

            if args.benign:
                aug_test_x, diff = benign_aug(test_x)
                label.append([test_label.numpy()])
            elif args.adv:
                aug_test_x, diff, feature_map_num, feature_idx = adv_aug.R_FGSM(test_x, test_label.cuda())
                label.append([test_label.item()] * 16)

            # print(aug_test_x[0,0,0]==aug_test_x[1,0,0])

            
            diff_array.append(diff)
            npy_adv_images.append(aug_test_x)
            feature_idx_array.append(feature_idx)


    

    diff_array = np.array(diff_array)
    label = np.squeeze(np.array(label))

    
    npy_adv_images = np.array(npy_adv_images)[:,:,0]
    n, b, c, w, h = npy_adv_images.shape 

    npy_adv_images = np.reshape(npy_adv_images, (n*b, c, w, h))    
    feature_idx_array = np.reshape(np.array(feature_idx_array), (n*b))
    diff_array = np.reshape(diff_array, n*b)
    label = np.reshape(label, n*b)

    # print(feature_idx_array)

    # img_save = np.concatenate([npy_adv_images[:,i] for i in range(b)], axis=0)
    # sorted_label = np.concatenate([label[:,i] for i in range(b)], axis=0)
    # print(sorted_label)

    
    idx = np.argsort(diff_array)
    img_save = []
    sorted_label = []
    sorted_feature_idx = []
    for i in idx:
        img_save.append(npy_adv_images[i])
        sorted_label.append(label[i])
        sorted_feature_idx.append(feature_idx_array[i])

    

    np.save(args.save_path + "full_sorted_cifar10_resnet20_all_img.npy", np.array(img_save))
    np.save(args.save_path + "full_sorted_cifar10_resnet20_all_label.npy", np.array(sorted_label))
    np.save('./full_sorted_feature_idx_all.npy', np.array(sorted_feature_idx))

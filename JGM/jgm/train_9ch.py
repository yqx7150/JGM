import numpy as np
import tqdm
from losses.dsm import anneal_dsm_score_estimation
import torch.nn.functional as F
import logging
import torch
import os
import shutil
import tensorboardX
import torch.optim as optim
from torchvision.datasets import MNIST, CIFAR10, SVHN
import torchvision.transforms as transforms
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset,Dataset
from models.cond_refinenet_dilated import CondRefineNetDilated
import glob
import cv2
import grad_image
from grad_image import *


__all__ = ['Train_9ch']

class LoadDataset(Dataset):
    def __init__(self,file_list,transform=None):
        self.files = glob.glob(file_list+ '/*.jpg')
        
        self.transform = transform
        self.to_tensor = T.ToTensor()
    def __len__(self):
        return len(self.files)
    def __getitem__(self,index):
        img = cv2.imread(self.files[index], 3)
        img = cv2.resize(img, (128, 128))
        im = img[:,:,:3]/255
        im_h = (get_h_input(im) + 1.0)/2.0
        im_v = (get_v_input(im) + 1.0)/2.0
        im = np.concatenate((im,im_h,im_v),2)
        '''print(im.shape,np.max(im),np.min(im))
        print(im_h.shape,np.max(im_h),np.min(im_h))
        print(im_v.shape,np.max(im_v),np.min(im_v))
        print(im.shape,np.max(im),np.min(im))
        assert False'''
        return im.transpose(2,0,1)

class Train_9ch():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))

    def logit_transform(self, image, lam=1e-6):
        image = lam + (1 - 2 * lam) * image
        return torch.log(image) - torch.log1p(-image)

    def train(self):
        if self.config.data.random_flip is False:
            tran_transform = test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])
        else:
            tran_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
            test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

        dataset = LoadDataset('./train_image/train2017',tran_transform)
        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=4)


        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)
        score = CondRefineNetDilated(self.config).to(self.config.device)

        score = torch.nn.DataParallel(score)

        optimizer = self.get_optimizer(score.parameters())

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint_145000.pth'))
            score.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        step = 145000

        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_classes))).float().to(self.config.device)

        for epoch in range(self.config.training.n_epochs):
            for i, X in enumerate(dataloader):
                X = torch.tensor(X,dtype=torch.float32)
                step += 1
                score.train()
                X = X.to(self.config.device)
                X = X / 256. * 255. + torch.rand_like(X) / 256.
                if self.config.data.logit_transform:
                    X = self.logit_transform(X)

                labels = torch.randint(0, len(sigmas), (X.shape[0],), device=X.device)
                if self.config.training.algo == 'dsm':
                    loss = anneal_dsm_score_estimation(score, X, labels, sigmas, self.config.training.anneal_power)
                elif self.config.training.algo == 'ssm':
                    loss = anneal_sliced_score_estimation_vr(score, X, labels, sigmas,
                                                             n_particles=self.config.training.n_particles)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tb_logger.add_scalar('loss', loss, global_step=step)
                logging.info("step: {}, loss: {}".format(step, loss.item()))

                if step >= self.config.training.n_iters:
                    return 0

                
                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))

                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))

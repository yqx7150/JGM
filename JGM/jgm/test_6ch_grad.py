import os
from natsort import natsorted
import cv2
import numpy as np
import torch
import torch.nn as nn
import glob
from models.cond_refinenet_dilated import CondRefineNetDilated
#from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from scipy.misc import imread,imresize
from skimage.measure import compare_psnr,compare_ssim 
from scipy.io import savemat
import grad_image
from grad_image import *

__all__ = ['Test_6ch_grad']

def write_Data(result_all,i):
    with open(os.path.join('./output/',"psnr"+".txt"),"w+") as f:
        for i in range(len(result_all)):
            f.writelines('current image {} PSNR : '.format(i) + str(result_all[i][0]) + \
            "    SSIM : " + str(result_all[i][1]))
            f.write('\n')

class Test_6ch_grad():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def test(self):
        # Load the score network
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        scorenet = CondRefineNetDilated(self.config).to(self.config.device)
        scorenet = torch.nn.DataParallel(scorenet)
        scorenet.load_state_dict(states[0])
        scorenet.eval()

        batch_size = 1 
        samples = 1
        image_count = 2

        files_list = glob.glob('./test/000091.png')
        files_list = natsorted(files_list)
        length = len(files_list)
        result_all = np.zeros([2,2])
        for z,file_path in enumerate(files_list):
            x0 = cv2.imread(file_path)
            x0 = cv2.resize(x0,(128,128))
            original_image = x0.copy()
            original_image = np.array(original_image,dtype=np.float32)
            x0 = torch.tensor(x0.transpose(2,0,1),dtype=torch.float).unsqueeze(0) / 255.0
            x_stack = torch.zeros([x0.shape[0]*samples,x0.shape[1],x0.shape[2],x0.shape[3]],dtype=torch.float32)

            for i in range(samples):
                x_stack[i*batch_size:(i+1)*batch_size,...] = x0
            x0 = x_stack           
            
            gray = ((x0[:,0,...] + x0[:,1,...] + x0[:,2,...])).cuda()/3.0
            gray_mixed = torch.stack([gray,gray,gray],dim=1)
            #gray_mixed = x0
            
            gray_mixed_h = np.zeros([1,128,128,3],dtype=np.float32)
            gray_mixed_v = np.zeros([1,128,128,3],dtype=np.float32)
            for i in range(gray_mixed.shape[0]):
                gray_mixed_h[i,...] = get_h_input(gray_mixed[i,...].cpu().detach().numpy().transpose(1,2,0))
                gray_mixed_v[i,...] = get_v_input(gray_mixed[i,...].cpu().detach().numpy().transpose(1,2,0))
            gray_mixed_hv = np.concatenate((gray_mixed_h,gray_mixed_v),3).transpose(0,3,1,2)
            
            x0 = nn.Parameter(torch.Tensor(samples*batch_size,6,x0.shape[2],x0.shape[3]).uniform_(-1,1)).cuda()
            x01 = x0.clone()

            step_lr=0.0003 * 0.005 # bedroom 0.1  church 0.2 
            sigmas = np.array([1., 0.59948425, 0.35938137, 0.21544347, 0.12915497, 0.07742637, 0.04641589, 0.02782559, 0.01668101, 0.01])
            n_steps_each = 100
            max_psnr = 0
            max_ssim = 0
            for idx, sigma in enumerate(sigmas):
                lambda_recon = 1./sigma**2
                labels = torch.ones(1, device=x0.device) * idx
                labels = labels.long()

                step_size = step_lr * (sigma / sigmas[-1]) ** 2
                
                print('sigma = {}'.format(sigma))
                for step in range(n_steps_each):
                    print('current step %03d iter' % step)                    
                    x0_mix = (x01[:,0,...] + x01[:,1,...] + x01[:,2,...] + x01[:,3,...] + x01[:,4,...] + x01[:,5,...])/6.0     
                    error = torch.stack([x0_mix,x0_mix,x0_mix,x0_mix,x0_mix,x0_mix],dim=1) - torch.tensor(gray_mixed_hv).cuda() 
                    noise_x = torch.randn_like(x01) * np.sqrt(step_size * 2)
                    grad_x0  = scorenet(x01, labels).detach()
                    x0 = x01 + step_size * (grad_x0 - lambda_recon * (error))
                    
                    aaa = x0[0,...].clone().detach().cpu().numpy().transpose(1,2,0)
                    cv2.imwrite('./output/x0.png',aaa[:,:,0:3].astype(np.uint8))
                    
                    x0_h = x0[:,0:3,:,:]
                    x0_v = x0[:,3:6,:,:]
                    x0_U = np.zeros([1,128,128,3],dtype=np.float32)

                    for i in range(x0.shape[0]):
                        x0_U[i,...] = from_grad_get_image(gray_mixed[i,...].cpu().detach().numpy().transpose(1,2,0),  x0_h[i,...].cpu().detach().numpy().transpose(1,2,0).clip(-1,1),  x0_v[i,...].cpu().detach().numpy().transpose(1,2,0).clip(-1,1),beta = 8.388608e+1 / 2.)
                    
                    #gray_mixed = x0_U.transpose(0,3,1,2)
                    #gray_mixed = torch.tensor(gray_mixed,dtype=torch.float32)/255.
                    
                    x01 = x0.clone() + noise_x                                                  
                    
                    max_result,post = 0,0

                    for i in range(x0_U.shape[0]):
                        psnr = compare_psnr(x0_U[i,...]*255.0,original_image,data_range=255)
                        ssim = compare_ssim(x0_U[i,...],original_image/255.0,data_range=1,multichannel=True)
                        if max_result < psnr:
                            max_result = psnr
                            post = i
                        print("current {} step".format(step),'PSNR :', psnr,'SSIM :', ssim)

                    #savemat('./Rec_Best',{'img':x_rec[post,...]})
                    if max_psnr < psnr :
                        result_all[z,0] = psnr
                        max_psnr = psnr
                        result_all[length,0] = sum(result_all[:length,0])/length
                        
                    if max_ssim < ssim:
                        result_all[z,1] = ssim
                        max_ssim = ssim
                        result_all[length,1] = sum(result_all[:length,1])/length
                    
                    write_Data(result_all,z)              
                   
            self.write_images(torch.tensor((x0_U).transpose(0, 3, 1, 2)).detach().cpu(), 'x_end_rgb.png',samples,z)

    def write_images(self, x,name,n=7,z=0):
        x = x.numpy().transpose(0, 2, 3, 1)
        d = x.shape[1]
        panel = np.zeros([1*d,n*d,3],dtype=np.uint8)
        for i in range(1):
            for j in range(n):
                panel[i*d:(i+1)*d,j*d:(j+1)*d,:] = (256*(x[i*n+j])).clip(0,255).astype(np.uint8)[:,:,::-1]
        cv2.imwrite(os.path.join(self.args.image_folder, 'img_{}_Rec_'.format(z) + name), panel)

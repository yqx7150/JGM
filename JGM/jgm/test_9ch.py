import os
import shutil
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
import grad_image
import matplotlib.pyplot as plt
from grad_image import *

__all__ = ['Test_9ch']

def write_Data(result_all_13,i,z):
    with open(os.path.join('./output/',"psnr_output_{}".format(z)+".txt"),"w+") as f:
        for i in range(len(result_all_13)):
            f.writelines('current image {} PSNR : '.format(i) + str(result_all_13[i][0]) + '    SSIM : ' + str(result_all_13[i][1]))
            f.write('\n')

def write_Data_finally(result_all,i):
    with open(os.path.join('./output/output/',"psnr_output_finally"+".txt"),"w+") as f:
        for i in range(len(result_all)):
            f.writelines('current image {} PSNR : '.format(i) + str(result_all[i][0]) + '    SSIM : ' + str(result_all[i][1]))
            f.write('\n')

class Test_9ch():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def test(self):
        # Load the score network
        states = torch.load(os.path.join(self.args.log, 'checkpoint_140000.pth'), map_location=self.config.device)   #coco checkpoint_140000
        scorenet = CondRefineNetDilated(self.config).to(self.config.device)
        scorenet = torch.nn.DataParallel(scorenet)
        scorenet.load_state_dict(states[0])
        scorenet.eval()
        aaa = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        batch_size = 1 
        samples = 1
        files_list = glob.glob('./24_i.png')
        files_list = natsorted(files_list)
        length = len(files_list)
        result_all_12 = np.zeros([12,2])
        result_all = np.zeros([101,2])
        for z,file_path in enumerate(files_list):
            for c in range(12):
                img = cv2.imread(file_path)
                img = cv2.resize(img,(128,128))
                
                original_image = img.copy()
                img = torch.tensor(img.transpose(2,0,1),dtype=torch.float32).unsqueeze(0) / 255.0
                x_stack = torch.zeros([img.shape[0]*samples,img.shape[1],img.shape[2],img.shape[3]],dtype=torch.float32)

                for i in range(samples):
                    x_stack[i*batch_size:(i+1)*batch_size,...] = img
                img = x_stack           
                gray = ((img[:,0,...] +  img[:,1,...] + img[:,2,...])).cuda()/3.0
                #gray = ((0.2126*img[:,0,...] +  0.7152*img[:,1,...] + 0.0722*img[:,2,...])).cuda()
                gray_mixed = torch.stack([gray,gray,gray],dim=1)

                u0_h = np.zeros([1,128,128,3],dtype=np.float32)
                u0_v = np.zeros([1,128,128,3],dtype=np.float32)
                for i in range(gray.shape[0]):
                    u0_h[i,...] = get_h_input(gray_mixed[i,...].cpu().detach().numpy().transpose(1,2,0))
                    u0_v[i,...] = get_v_input(gray_mixed[i,...].cpu().detach().numpy().transpose(1,2,0))
                u0_h_mixed = torch.tensor(u0_h.transpose(0,3,1,2) +1.)/2.
                u0_v_mixed = torch.tensor(u0_v.transpose(0,3,1,2) +1.)/2.

                x0 = nn.Parameter(torch.Tensor(samples*batch_size,9,img.shape[2],img.shape[3]).uniform_(-1,1)).cuda()
                x01 = x0.clone()

                step_lr=0.00003 * 0.09    #coco 0.0000599 * 0.26
                #step_lr_grad=0.0003 * 0.00125
                sigmas = np.array([1., 0.59948425, 0.35938137, 0.21544347, 0.12915497,0.07742637, 0.04641589, 0.02782559, 0.01668101, 0.01]) 
                
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
                        x0_mix = (x01[:,0,...] + x01[:,1,...] + x01[:,2,...])/3.0
                        x0_mix_grad_h = (x01[:,3,...] + x01[:,4,...] + x01[:,5,...])/3.0
                        x0_mix_grad_v = (x01[:,6,...] + x01[:,7,...] + x01[:,8,...])/3.0
                        '''x0_mix = (0.2126*x01[:,0,...] + 0.7152*x01[:,1,...] + 0.0722*x01[:,2,...])
                        x0_mix_grad_h = (0.2126*x01[:,3,...] + 0.7152*x01[:,4,...] + 0.0722*x01[:,5,...])
                        x0_mix_grad_v = (0.2126*x01[:,6,...] + 0.7152*x01[:,7,...] + 0.0722*x01[:,8,...])'''
                        
                        error = torch.stack([x0_mix,x0_mix,x0_mix],dim=1).to(device) - gray_mixed.to(device)
                        error_grad_h = torch.stack([x0_mix_grad_h,x0_mix_grad_h,x0_mix_grad_h],dim=1).to(device) - u0_h_mixed.to(device)
                        error_grad_v = torch.stack([x0_mix_grad_v,x0_mix_grad_v,x0_mix_grad_v],dim=1).to(device) - u0_v_mixed.to(device)

                        noise_x = torch.randn_like(x01) * np.sqrt(step_size * 2)

                        grad_x0 = scorenet(x01, labels).detach()
                     
                        x0 = x01 + step_size * (grad_x0)
                        x0[:,0:3,...] = x0[:,0:3,...] - step_size * lambda_recon * error
                        x0[:,3:6,...] = x0[:,3:6,...] - step_size * lambda_recon * error_grad_h
                        x0[:,6:9,...] = x0[:,6:9,...] - step_size * lambda_recon * error_grad_v

                        #x0 = torch.mean(x0,dim=0)
                        #x0 = torch.stack([x0,x0,x0,x0,x0,x0,x0,x0],dim=0)

                        x0_image = x0[:,0:3,...]
                        x0_grad_h = x0[:,3:6,...]
                        x0_grad_v = x0[:,6:9,...]
                        
                        x0_U = np.zeros([1,128,128,3],dtype=np.float32)
                        for i in range(x0.shape[0]):
                            x0_U[i,...] = from_grad_get_image(x0_image[i,...].cpu().detach().numpy().transpose(1,2,0),  ((x0_grad_h[i,...] * 2.0) - 1.0).cpu().detach().numpy().transpose(1,2,0).clip(-1,1),  ((x0_grad_v[i,...] * 2.0) - 1.0).cpu().detach().numpy().transpose(1,2,0).clip(-1,1), beta = 5.9)  
                                            
                        x01 = x0.clone().to(device) + noise_x.to(device)
                       	
                        x_rec = x0_U
                        original_image = np.array(original_image,dtype = np.float32)

                        for i in range(x_rec.shape[0]):
                            psnr = compare_psnr(x_rec[i,...]*255.0,original_image,data_range=255)
                            ssim = compare_ssim(x_rec[i,...],original_image/255.0,data_range=1,multichannel=True)
                            print("current {} step".format(step),'PSNR :', psnr,'SSIM :', ssim)
                            
                        if max_psnr < psnr :
                            result_all_12[c,0] = psnr
                            max_psnr = psnr
                            #cv2.imwrite(os.path.join(self.args.image_folder, 'img_{}_Rec_9ch_finally.png'.format(z)),(x_rec[i,...]*256.0).clip(0,255).astype(np.uint8))
                            #result_all_12[length,0] = sum(result_all_12[:length,0])/length
                        if max_ssim < ssim:
                            result_all_12[c,1] = ssim
                            max_ssim = ssim
                            #result_all_12[length,1] = sum(result_all_12[:length,1])/length
                        write_Data(result_all_12,c,z)
                    
                x_save = x0_U
                x_save_R = x_save[:,:,:,2:3]
                x_save_G = x_save[:,:,:,1:2]
                x_save_B = x_save[:,:,:,0:1]
                x_save = np.concatenate((x_save_R,x_save_G,x_save_B),3)
                self.write_images(torch.tensor(x_save.transpose(0,3,1,2)), 'x_end_rgb_{}.png'.format(c),samples,z)
                result_all[z,:] = result_all_12[np.argmax(result_all_12[:,0],0),:]
            files_list_12 = glob.glob('./output/img_{}_Rec_9ch_x_end_rgb_{}.png'.format(z,np.argmax(result_all_12[:,0],0)))
            shutil.copy(files_list_12[0],'./output/output/')
            result_all[100,:]=np.sum(result_all[0:100,:],0)/100
            write_Data_finally(result_all,z)
            
    def write_images(self, x,name,n=1,z=0):
        x = x.numpy().transpose(0, 2, 3, 1)
        d = x.shape[1]
        panel = np.zeros([1*d,n*d,3],dtype=np.uint8)
        for i in range(1):
            for j in range(n):
                panel[i*d:(i+1)*d,j*d:(j+1)*d,:] = (256*(x[i*n+j])).clip(0,255).astype(np.uint8)[:,:,::-1]

        cv2.imwrite(os.path.join(self.args.image_folder, 'img_{}_Rec_9ch_'.format(z) + name), panel)#

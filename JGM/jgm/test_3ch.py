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

__all__ = ['Test_3ch']

def write_Data(result_all,i):
    with open(os.path.join('./rgb_output_ccc/',"psnr"+".txt"),"w+") as f:
        for i in range(len(result_all)):
            f.writelines('current image {} PSNR : '.format(i) + str(result_all[i][0]) + \
            "    SSIM : " + str(result_all[i][1]))
            f.write('\n')

class Test_3ch():
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

        files_list = glob.glob('./test/*.png')
        files_list = natsorted(files_list)
        length = len(files_list)
        result_all = np.zeros([999,2])
        for z,file_path in enumerate(files_list):
            x0 = cv2.imread(file_path)
            x0 = cv2.resize(x0,(128,128))
            original_image = x0.copy()
            x0 = torch.tensor(x0.transpose(2,0,1),dtype=torch.float32).unsqueeze(0) / 255.0
            x_stack = torch.zeros([x0.shape[0]*samples,x0.shape[1],x0.shape[2],x0.shape[3]],dtype=torch.float32)

            for i in range(samples):
                x_stack[i*batch_size:(i+1)*batch_size,...] = x0
            x0 = x_stack           
            
            gray = ((x0[:,0,...] + x0[:,1,...] + x0[:,2,...])).cuda()/3.0
            print(gray.shape)
            gray_mixed = torch.stack([gray,gray,gray],dim=1)
            print(gray_mixed.shape)

            x0 = nn.Parameter(torch.Tensor(samples*batch_size,3,x0.shape[2],x0.shape[3]).uniform_(-1,1)).cuda()
            x01 = x0.clone()

            x0_mix = ((x01[:,0,...] + x01[:,1,...] + x01[:,2,...]))/3.0

            recon = (torch.stack([x0_mix,x0_mix,x0_mix],dim=1) - gray_mixed)**2

            step_lr=0.00003 * 0.1 # bedroom 0.1  church 0.2 
            sigmas = np.array([1., 0.59948425, 0.35938137, 0.21544347, 0.12915497, 0.07742637, 0.04641589, 0.02782559, 0.01668101, 0.01])
            n_steps_each = 60
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
                    x0_mix = ((x01[:,0,...] + x01[:,1,...] + x01[:,2,...]))/3.0                    
                    error = torch.stack([x0_mix,x0_mix,x0_mix],dim=1) - gray_mixed 
                    noise_x = torch.randn_like(x01) * np.sqrt(step_size * 2)
                    grad_x0  = scorenet(x01, labels).detach()
                    x0 = x01 + step_size * (grad_x0 - lambda_recon * (error))   # + noise_x
                    x01 = x0.clone() + noise_x                    
                    
                    x_rec = x0.clone().detach().cpu().numpy().transpose(0,2,3,1)#.clip(0,1)
                    max_result,post = 0,0
                    original_image = np.array(original_image,dtype=np.float32)
                    for i in range(x_rec.shape[0]):
                        psnr = compare_psnr(x_rec[i,...]*255.0,original_image,data_range=255)
                        ssim = compare_ssim(x_rec[i,...],original_image/255.0,data_range=1,multichannel=True)
                        if max_result < psnr:
                            max_result = psnr
                            post = i
                        print("current {} step".format(step),'PSNR :', psnr,'SSIM :', ssim)

                    savemat('./Rec_Best',{'img':x_rec[post,...]})
                    if max_psnr < psnr :
                        result_all[z,0] = psnr
                        max_psnr = psnr
                        result_all[length,0] = sum(result_all[:length,0])/length
                        
                    if max_ssim < ssim:
                        result_all[z,1] = ssim
                        max_ssim = ssim
                        result_all[length,1] = sum(result_all[:length,1])/length
                    
                    write_Data(result_all,z)              
 
            x_save = x0.clone().detach().cpu()
            x_save = np.array(x_save)
            x_save_R = x_save[:,2:3,:,:]
            x_save_G = x_save[:,1:2,:,:]
            x_save_B = x_save[:,0:1,:,:]
            x_save = np.concatenate((x_save_R,x_save_G,x_save_B),1)
            self.write_images(torch.tensor(x_save).detach().cpu(), 'x_end_rgb.png',samples,z)

    def write_images(self, x,name,n=7,z=0):
        x = x.numpy().transpose(0, 2, 3, 1)
        d = x.shape[1]
        panel = np.zeros([1*d,n*d,3],dtype=np.uint8)
        for i in range(1):
            for j in range(n):
                panel[i*d:(i+1)*d,j*d:(j+1)*d,:] = (256*(x[i*n+j])).clip(0,255).astype(np.uint8)[:,:,::-1]
        cv2.imwrite(os.path.join(self.args.image_folder, 'img_{}_Rec_'.format(z) + name), panel)

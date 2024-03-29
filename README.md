# JGM

**Paper**: Joint Intensity-Gradient Guided Generative Modeling for Colorization

**Authors**: Kai Hong, Jin Li, Wanyun Li, Cailian Yang, Minghui Zhang, Yuhao Wang, Qiegen Liu,


Date : 12/2020
Version : 1.0   
The code and the algorithm are for non-comercial use only. 
Copyright 2020, Department of Electronic Information Engineering, Nanchang University.  

This paper proposes an iterative generative model for solving the automatic colorization problem. Although pre-vious researches have shown the capability to generate plausible color, the edge color overflow and the require-ment of the reference images still exist. The starting point of the unsupervised learning in this study is the observation that the gradient map possesses latent information of the image. Therefore, the inference process of the generative modeling is conducted in joint intensity-gradient domain. Specifically, a set of intensity-gradient formed high-dimensional tensors, as the network input, are used to train a powerful noise conditional score network at the training phase. Furthermore, the joint intensity-gradient constraint in data-fidelity term is proposed to limit the de-gree of freedom within generative model at the iterative colorization stage, and it is conducive to edge-preserving. Extensive experiments demonstrate that the system outper-forms state-of-the-art methods whether in quantitative comparisons or user study.
## Training
if you want to train the code, please train the code to attain the model

```bash 
python3.5 JGM_main.py --jgm Train_9ch --config anneal.yml --doc your save path
```

## Test
if you want to test the code, please 

```bash 
python3.5 JGM_main.py --igm Test_9ch --config anneal.yml --doc your checkpoint --test --image_folder your save path
```


## Checkpoints
We provide pretrained checkpoints. You can download pretrained models from [Baidu Drive](https://pan.baidu.com/s/16eTJYctxh6t3mkg-fZvLdw). 
key number is "JGM9 " 
## Graphical representation
## Visualization of the colorization results with different generative modelings.
 <div align="center"><img src="https://github.com/yqx7150/JGM/blob/main/image/1.png" >  </div>
 Visualization of the colorization results with different generative modelings. (a) The reference grayscale image. (b) The top line is the colorization result with generative modeling in intensity domain, and the bottom line is with the generative modeling in joint intensity-gradient do-main. (c) The colorization result of the proposed JGM. Particularly, the generative modeling of JGM is conducted in 9-channel (intensity-gradient) domain, largely reduces the color ambiguity in intensity and attains a more natural and realistic result.
 
 ##  The visual comparison of CVAE , iGM  and JGM
 <div align="center"><img src="https://github.com/yqx7150/JGM/blob/main/image/3.1.jpg" >  </div>
 Visual comparison of CVAE (a), iGM (b) and JGM (c). Rather than the CVAE that optimally encode the compressible latent space to achieve the colorization goal, both iGM and JGM utilize the generative modeling in high-dimensional space to optimize the colorization target. Particularly by taking advantage of the joint intensity-gradient field, the proposed JGM learns prior information and iteratively approaches to color image.
 
  ##   The pipeline of the prior learning stage and the iterative colorization procedure of JGM.
 <div align="center"><img src="https://github.com/yqx7150/JGM/blob/main/image/3.png" >  </div>
 The pipeline of the prior learning stage and the iterative colorization procedure of JGM. More specifically, the prior training stage learns the data distribution (including images domain and gradients domain) from the reference dataset, which acts as prior information for later colorization. The colorization stage generates samples from the high-dimensional noisy data distribution by annealed Langevin dynamics, under the given intensity-gradient data-consistency constraint. 
 
   ##    Visualization of the intermediate colorization process with annealed Langevin dynamics.
 <div align="center"><img src="https://github.com/yqx7150/JGM/blob/main/image/4.png" >  </div>
 Visualization of the intermediate colorization process with annealed Langevin dynamics. As the level of artificial noise becomes smaller, the colori-zation results tend to more natural color effects.
 
 ##    Visual comparisons with the state-of-the-arts.
 <div align="center"><img src="https://github.com/yqx7150/JGM/blob/main/image/5.png">  </div>
 Visual comparisons with the state-of-the-arts. From left to right: Grayscale, Zhang et al., MemoPainter, ChromaGAN, iGM and JGM. Our method with gradient domain and high-dimensional can predict pleasing colors visually.
 
 ##    Diversified colorization.
 <div align="center"><img src="https://github.com/yqx7150/JGM/blob/main/image/6.png">  </div>
 Diversified colorization effects of the proposed JGM.
 
## Train Data
We choose three datasets for experiments, including LSUN(bedroom and church), COCO-stuff and ImageNet 
## Test Data
We randomly select 100 bedrooms and church data respectively, the size is 128x128.

### Other Related Projects

  * Homotopic Gradients of Generative Density Priors for MR Image Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/9435335)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/HGGDP)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)   [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

 * Iterative Reconstruction for Low-Dose CT using Deep Gradient Priors of Generative Model  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/9703672)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EASEL)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)

  * Universal Generative Modeling for Calibration-free Parallel MR Imaging  
[<font size=5>**[Paper]**</font>](https://biomedicalimaging.org/2022/)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/UGM-PI)   [<font size=5>**[Poster]**</font>](https://github.com/yqx7150/UGM-PI/blob/main/paper%20%23160-Poster.pdf)   
     
 * Progressive Colorization via Interative Generative Models  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/9258392)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/iGM)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)   [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)


 * Wavelet Transform-assisted Adaptive Generative Modeling for Colorization
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/9782538)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/WACM)   [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

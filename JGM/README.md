# iGM

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
python3 JGM_main.py --jgm JGM_train_9ch --config anneal.yml --doc your save path
```

## Test
if you want to test the code, please 

```bash 
python3 JGM_main.py --igm Test_9ch --config anneal.yml --doc your checkpoint --test --image_folder your save path
```


## Checkpoints
We provide pretrained checkpoints. You can download pretrained models from [Baidu Drive](https://pan.baidu.com/s/1do-Y-13E7NWK2mkE9K912w). 
key number is "JGM " 

## Train Data
We choose three datasets for experiments, including LSUN(bedroom and church), COCO-stuff and ImageNet 
## Test Data
We randomly select 100 bedrooms and church data respectively, the size is 128x128.

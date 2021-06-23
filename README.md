# BicycleGAN
This project includes a well written blog on [BicycleGAN](/Blog.ipynb) and also its implementation in pytorch. It also includes the [results](BicycleGAN/Results/
) that I got after experimenting the model on two different datasets **Edges to Shoes** and **CIFAR10**

## Abstract
Many image-to-image translation problems are ambiguous, as a single input image may correspond to multiple possible outputs. In this work, we aim to model a distribution of possible outputs in a conditional generative modeling setting. The ambiguity of the mapping is distilled in a low-dimensional latent vector, which can be randomly sampled at test time. A generator learns to map the given input, combined with this latent code, to the output. We explicitly encourage the connection between output and the latent code to be invertible. This helps prevent a many-to-one mapping from the latent code to the output during training, also known as the problem of mode collapse, and produces more diverse results. We explore several variants of this approach by employing different training objectives, network architectures, and methods of injecting the latent code. Our proposed method encourages bijective consistency between the latent encoding and output modes. We present a systematic comparison of our method and other variants on both perceptual realism and diversity.

## Reference

```
@misc{zhu2018multimodal,
      title={Toward Multimodal Image-to-Image Translation}, 
      author={Jun-Yan Zhu and Richard Zhang and Deepak Pathak and Trevor Darrell and Alexei A. Efros and Oliver Wang and Eli Shechtman},
      year={2018},
      eprint={1711.11586},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Structure
.
├── BicycleGAN
│   ├── Denoising_CIFAR10.PY
│   ├── Results
│   │   ├── CIFAR10
│   │   └── Edges2Shoes
│   ├── bicyclegan.py
│   ├── datasets.py
│   ├── lpips.ipynb
│   └── models.py
├── Blog.ipynb
├── README.md
├── data
│   └── download_pix2pix_dataset.sh
├── img
├── requirements.txt
└── tree.txt

#### Installation
    $ git clone https://github.com/sahilg06/BicycleGAN.git
    $ cd BicycleGAN/
    $ sudo pip3 install -r requirements.txt
    
    
#### Run Example For  edges → images task
```
$ cd data/
$ bash download_pix2pix_dataset.sh edges2shoes
$ cd ../BicycleGAN/
$ python3 bicyclegan.py
```

#### Run Example For denoising task
```
$ cd BicycleGAN/
$ python3 Denoising_CIFAR10.PY
```
#### Analyse the results using Learned Perceptual Image Patch Similarity ([LPIPS](https://github.com/richzhang/PerceptualSimilarity)) metric [here](/BicycleGAN/lpips.ipynb)

# DDLoc localization: a sim-to-real learning method for absolute localization
This repo contains the Pytorch implementation of DDLoc, an adapation of ARC method for absolute coordinate regression.

Please make sure you have access to the **CrossLoc Benchmark Raw Datasets** and have set it up properly before proceeding. 

Also check out other useful repos regarding the datasets:

*  [**`CrossLoc-Benchmark-Datasets`**](https://github.com/TOPO-EPFL/CrossLoc-Benchmark-Datasets): CrossLoc benchmark datasets setup tutorial.
* [**`TOPO-DataGen`**](https://github.com/TOPO-EPFL/TOPO-DataGen): an open and scalable aerial synthetic data generation workflow.

The DDLoc localization algorithm is officially presented in the paper
<br>
**CrossLoc: Scalable Aerial Localization Assisted by Multimodal Synthetic Data**
<br>
[Qi Yan](https://qiyan98.github.io/), [Jianhao Zheng](https://jianhao-zheng.github.io/), [Simon Reding](https://people.epfl.ch/simon.reding/?lang=en), [Shanci Li](https://people.epfl.ch/shanci.li/?lang=en), [Iordan Doytchinov](https://people.epfl.ch/iordan.doytchinov?lang=en) 
<br>
École Polytechnique Fédérale de Lausanne (EPFL)
<br>
Links: **[arXiv](https://arxiv.org/abs/2112.09081) | [code repos](https://github.com/TOPO-EPFL/CrossLoc)**

Happy coding! :)

## Contents

- [Requirments](#requirements)
- [Training Precedures](#training-precedures)
- [Evaluations](#evaluations)
- [Pretrained Models](#pretrained-models)


## Requirements
1. Python 3.6 with Ubuntu 16.04
2. Pytorch 1.1.0
3. [dsacstar](https://github.com/vislearn/dsacstar) (if you want to test the camera pose estimation from the scene coordinate prediction)

You also need other third-party libraries, such as numpy, pillow, torchvision, and tensorboardX (optional) to run the code. 

We suggest to follow the procedure in [CrossLoc repo](https://github.com/TOPO-EPFL/CrossLoc) to install dependecies.

## Datasets
You have to download our provided urbanscape data and place them in the following structure to load the data.
####  Dataset Structure
```
urban (real)
    | train
        | rgb
        | poses
        | init
        | calibration
    | test
        | rgb
        | poses
        | init
        | calibration
urban (synthetic)
    | train
        | rgb
        | poses
        | init
        | calibration      
```
You can download naturescape data for more experiments and follow the same structure. 
## Training Precedures
- [1 Train Initial Coordinate Regressor C (train_C.py)](#1-Train-Initial-Coordinate-Regressor-C)
- [2 Train Style Translator T (train_T.py)](#2-Train-Style-Translator-T)
- [3 Train Initial Attention Module A (train_A.py)](#3-Train-Initial-Attention-Module-A)
- [4 Train Inpainting Module I (train_U.py)](#4-Train-Inpainting-module-I)
- [5 Jointly Train Coordinate Regressor C and Attention Module A (train_joint_A_C.py)](#5-Jointly-Train-Coordinate-Regressor-C-and-Attention-Module-A)
- [6 Finetune the Coordinate Regressor C with translated image (train_finetune_C.py)](#6-Finetune-the-Coordinate-Regressor-C-with-translated-image)

We provide example scripts for training each step in [this folder](./scripts/train)
`batch_size` and `eval_batch_size` are flexible to change given your working environment.

#### 1 Train Initial Coordinate Regressor C
Train an initial coordinate regressor C with real and synthetic data. The best model is picked by the one with minimum camera poses error. The checkpoints are saved in `./checkpoints/your_dir_name/train_initial_coord_regressor_C/`.
#### 2 Train Style Translator T
Train the style translator T with naive mixed data and finetune T by paired real and synthetic data. The best model is picked by visual inspection & training loss curves. 
#### 3 Train Initial Attention Module A 
Train an initial attention module A from scratch with descending $\tau$ values.
#### 4 Train Inpainting Module I
Train the inpainting module I with T (from step 2) and A (from step 3). 
#### 5 Jointly Train Coordinate Regressor C and Attention Module A
Further jointly train coordinate regressor C and attention module A together with C (from step 1), T (from step 2), A (from step 3) and I (from step 4). The A and C learned from this step is the good initialization before finetuning C with coordinate regression loss and reprojecetion loss. In step 5, we train for relatively less epochs. 
#### 6 Finetune the Coordinate Regressor C with translated image
Lastly, we finetune the coordinate regressor C with oordinate regression loss and reprojecetion loss using C (from step 5). The training translated image is generated by T (from step 2), A (from step 5) and I (from step 4). [generate_translated.py](./generate_translated.py) can be used to generate translated images from real images. An example is given in [Generate_translated.sh](./scripts/Generate_translated.sh)

## Evaluations
Evaluate the final results, you can make use of [eval.py](./eval.py) with an example given in [eval.sh](./scripts/eval.sh)
If you want to evaluate with your own data, please place your own data under `<real dataset>/test` with the dataset structure described above.

## Pretrained Models
The pretrained models will be released soon.

## Sample Result Visualization

<p align="center"><img src="imgs/urban.gif" alt="cf_land" width="600"/></p>
<p align="center"><img src="imgs/nature.gif" alt="cf_land" width="600"/></p>
<!-- <img src='imgs/urban.gif' align="left" width=420>
<img src='imgs/nature.gif' align="right" width=420> -->

<br><br><br>

## Acknowledgments
This code is developed based on [ARC](https://github.com/yzhao520/ARC) and [Pytorch-CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

## Citation

If you find our code useful for your research, please cite the paper:

````bibtex
@article{yan2021crossloc,
  title={CrossLoc: Scalable Aerial Localization Assisted by Multimodal Synthetic Data},
  author={Yan, Qi and Zheng, Jianhao and Reding, Simon and Li, Shanci and Doytchinov, Iordan},
  journal={arXiv preprint arXiv:2112.09081},
  year={2021}
}
````

# ReDO: Unsupervised Object Segmentation by Redrawing

Code for paper [Unsupervised Object Segmentation by Redrawing](https://arxiv.org/abs/1905.13539) by Mickaël Chen, Thierry Artières and Ludovic Denoyer. Presented as poster at NeurIPS 2019, Vancouver.

![redo](https://github.com/mickaelChen/ReDO/blob/master/imgs/redo.png)

We discover meaningful segmentation masks by redrawing regions of the images independently.

## Table of Contents

- [Random samples](#random-samples)
  * Samples for Flowers, LFW, CUB and toy dataset
  * A more diverse dataset with two classes
- [Datasets instructions](#datasets-instructions)
  * Flowers
  * CUB
  * LFW
- [Usage](#usage)
  * Pretrained models
  * Training ReDO

## Random samples

### Samples for Flowers, LFW, CUB and toy dataset
![samples](https://github.com/mickaelChen/ReDO/blob/master/imgs/redo_samples.png)

### A more diverse dataset with two classes

During the rebuttal process, we were asked to demonstrate that ReDO can work when the dataset contains multiple classes.
We build a new dataset by combining LFW and Flowers images (without labels). This new dataset has more variability,
contains different types of objects, and display a more obvious correlation between the object and the background. 
We trained ReDO without further hyperparameter tuning (not optimal), and obtained a reasonable accuracy of 0.856 and IoU of 0.691.
![lfw + flowers](https://github.com/mickaelChen/ReDO/blob/master/imgs/redo_lfwxflowers.png)

## Datasets instructions

### Flowers
1. Download and extract: *Dataset*, *Segmentations*, and *data splits* from http://www.robots.ox.ac.uk/~vgg/data/flowers/102/ 
2. The obtained *jpg* folder, *segmin* folder and *setid.mat* file should be placed in the same data root folder.

### CUB
1. Download and extract *Images* and *Segmentations* from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html 
2. Place the *segmentations* folder in the *CUB_200_2011/CUB_200_2011* subfolder.
3. Place the *train_val_test_split.txt* file from this repo in the *CUB_200_2011/CUB_200_2011* subfolder.
4. dataroot should be set to the *CUB_200_2011/CUB_200_2011* subfolder.

### LFW
1. Download and extract the *funneled images* from http://vis-www.cs.umass.edu/lfw/
2. Download and extract the *ground truth images* from http://vis-www.cs.umass.edu/lfw/part_labels/
3. Place the obtained *lfw_funneled* and *parts_lfw_funneled_gt_images* folders in the same data root folder.
4. Also place the *train.txt*, *val.txt* and *test.txt* files from the repo in this data root folder.


## Usage

Tested on python3.7 with pytorch 1.0.1

### Load pretrained models
Weights pretrained on Flowers, LFW, and CUB datasets can be downloaded from [google drive](https://drive.google.com/drive/folders/1hUb2iOTJAbWw1NotWGAsEt4ASomhOwbh).

- *dataset_nets_state.tar.gz*: pretrained weights for all 4 networks used during training in a single file.

The weights for the individual networks are also available, for instance if you only need to segment and/or redraw:

- *dataset_netM_state.tar.gz*: pretrained weights for mask extractor only. Enough if interested only in segmentation.
- *dataset_netX_state.tar.gz*: pretrained weights for region generators. Used to redraw objects.
- *dataset_netD_state.tar.gz*: pretrained weights for discriminator.
- *dataset_netZ_state.tar.gz*: pretrained weights for the network that infer the latent code z from image.

*.tar.gz* archives have to be uncompressed first to recover the *.pth* files containing the weights.

Provided example script needs at least netM and netX and is used as follows:

If using *dataset_nets_state.pth* on GPU cuda device 0

```
python example_load_pretrained.py --statePath path_to_nets_state.pth --dataroot path_to_data --device cuda:0
```

If using *dataset_netX_state.pth* and *dataset_netM_state.pth* on cpu:
```
python example_load_pretrained.py --statePathX path_to_netX_state.pth --statePathM path_to_netM_state.pth --dataroot path_to_data --device cpu
```


### Training from scratch

Examples:

```
python train.py --dataset flowers --nfX 32 --useSelfAttG --useSelfAttD --outf path_to_output_folder --dataroot path_to_data_folder --clean
```
```
python train.py --dataset lfw --useSelfAttG --useSelfAttD --outf path_to_output_folder --dataroot path_to_data_folder --clean
```

Some clarifications about the training process and the collapse issue:

As mentionned in the paper, the model can collapse with one region taking the whole image.
This happens early in the training (at about 3-5k iterations) in some runs (about 3.5 out of 10 in my experiments).
In this case, it is possible to restart training automatically using option --autoRestart .15 (or smaller for lfw).

After these early stages, training should be stable.
I stop training in the 20k~40k range, but the model gets unstable again if you train for too long.

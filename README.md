# ReDO

Code for paper [Unsupervised Object Segmentation by Redrawing](https://arxiv.org/abs/1905.13539).


![redo](https://github.com/mickaelChen/ReDO/blob/master/imgs/redo.png)

## Datasets

### Flowers:
1. Download and extract: Dataset, Segmentations, and data splits from http://www.robots.ox.ac.uk/~vgg/data/flowers/102/ 
2. The obtained jpg folder, segmin folder and setid.mat file should be placed in the same folder.

### CUB:
1. Download and extract Images and Segmentations from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html 
2. Place the segmentations folder in the CUB_200_2011 folder.
3. Place the train_val_test_split.txt file from this repo in the CUB_200_2011 folder.
4. dataroot should be set to the CUB_200_2011 folder.

### LFW:
1. Download and extract the funneled images from http://vis-www.cs.umass.edu/lfw/
2. Download and extract the ground truth images from http://vis-www.cs.umass.edu/lfw/part_labels/
3. Place the obtained lfw_funneled and parts_lfw_funneled_gt_images folders in the same folder.
4. Place the train.txt, val.txt and test.txt files from this repo in the same folder.


## Example usage

Tested on python3.7 with pytorch 1.0.1

```
python redo.py --dataset flowers --nfX 32 --useSelfAttG --useSelfAttD --outf path_output --dataroot path_to_data
```

## Random samples (from paper)
Those are not cherry-picked.

Column 1: Input

Column 2: Ground Truth

Column 3: output mask for object 1

Columns 4-7: generated image with redrawn object 1

Columns 8-11: generated image with redrawn object 2

![flowers](https://github.com/mickaelChen/ReDO/blob/master/imgs/flowers.png)
![lfw](https://github.com/mickaelChen/ReDO/blob/master/imgs/lfw.png)
![cub](https://github.com/mickaelChen/ReDO/blob/master/imgs/cub.png)
![c2mnist](https://github.com/mickaelChen/ReDO/blob/master/imgs/cmnist.png)

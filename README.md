# ReDO

Code for paper Unsupervised Object Segmentation by Redrawing (arXiv incoming)

![redo](https://github.com/mickaelChen/ReDO/blob/master/imgs/redo.png)

## Datasets
Flowers: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/
CUB-200: http://www.vision.caltech.edu/visipedia/CUB-200.html
LFW-funneled: http://vis-www.cs.umass.edu/lfw/
segmentations for LFW-funneled : http://vis-www.cs.umass.edu/lfw/part_labels/


## Example usage

Tested on python3.7 with pytorch 1.0.1

datasplits files are to be put directly in the dataset folder

```
python redo.py --nfX 32 --useSelfAttG --useSelfAttD --outf path_output --dataroot path_to_data
```

## Random samples (from paper)
Those are not cherry-picked

![flowers](https://github.com/mickaelChen/ReDO/blob/master/imgs/flowers.png)
![lfw](https://github.com/mickaelChen/ReDO/blob/master/imgs/lfw.png)
![cub](https://github.com/mickaelChen/ReDO/blob/master/imgs/cub.png)
![c2mnist](https://github.com/mickaelChen/ReDO/blob/master/imgs/cmnist.png)

## Introduction
This repo is an unofficial implementation of [IoU Loss for 2D/3D Object Detection](https://arxiv.org/pdf/1908.03851.pdf). It contains the simple calculattion of IoUs of 2D / 3D rotated bounding box.

## Requirements
Following dependencies are needed

    cudatoolkit=10.2
    pytorch>1.5         
    numpy
    matplotlib

## Usage

First, compile the CUDA extension.

    cd cuda_op
    python setup.py install

Then, run a demo which validate the Pytorch functions and CUDA extension.

    cd ..
    python IoUDemo.py
    
## Future Work
Complete the implementation of [GIoU-loss](https://giou.stanford.edu/GIoU.pdf) and [DIoU-loss](https://arxiv.org/abs/1911.08287) for rotated bounding boxes.




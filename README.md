# <p align ="center">Original-UNET</p>
![unet_orginal](https://user-images.githubusercontent.com/36896102/138446144-99c522c9-fd46-43e9-923f-3471f138e2b9.PNG)

This repo is made to implement original UNET as per paper 
"U-Net: Convolutional Networks for Biomedical Image Segmentation" by Ronneberger et al.
The modern variation of UNET padded convolution to make output mask same as input image
but this paper crops the input image with assumption that mostly mask is positioned at center of image.
"Due to the unpadded convolutions, the output image is smaller than the input
by a constant border width." -- quote from paper
While inference, input image need to be cropped as per mask size. 

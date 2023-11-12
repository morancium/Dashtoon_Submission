# Submission for **Dastoon** Problem Statement for **Engineer, Generative AI**
Here is the link of the [problem statement](https://dashtoon.notion.site/Engineer-Generative-AI-ca79a87773e54d769436cec05282e11e)


PS in brief:
> The aim of this assignment is to create a deep learning model capable of adapting an existing work to resemble the aesthetic of any art. The model should be able to analyze the artistic style of the selected art and apply similar stylistic features to a new, original artwork, creating a piece that seems as though it could have been created by the artist themselves.

The problem statement is closely related to the well known task called **Neural style transfer** aka **NST**

Description of NST:
> Neural style transfer (NST) refers to a class of software algorithms that manipulate digital images, or videos, in order to adopt the appearance or visual style of another image.

Similarly, for this problem statement, we were asked to apply the Artestic style of selected art (Style image) and apply similar stylistic feature to a new Image (Content image)

# **NST - Neural Style Transfer**

## Introduction
Neural Style Transfer (NST) is a fascinating application of deep learning that combines the content of one image with the style of another. This technique has gained popularity for creating artistic and visually appealing images. In this report, we will explore the implementation of NST using the VGG19 model.

Examples:
>![Dog and Painting](Images/NST1.png)
>![House and Painting](Images/NST2.png)
>![Myself and Mona lisa](Images/NST3.png)

## **Overview**
NST leverages the representations learned by a convolutional neural network (CNN) to separate and recombine content and style from different images. The basic idea is to define a content loss and a style loss, which are then minimized to generate a new image that preserves the content of one image and the style of another.


In this Jupyter notebook, I have tried implementing first paper on NST: **A Neural Algorithm of Artistic Style** by *Leon Gatys et al.* with minor changes and imporvements to enchance the output of the model

To make the model faster and more accurate, a **pre-trained** **VGG-Net-19** (Visual Geometry Group) is used. The model is trained on ImageNet images and can be downloaded from Pytorch API.

Some important Distinction for the understanding:




*   A content image (c) — the image we want to transfer a style to
*   A style image (s) — the image we want to transfer the style from
*   An input (generated) image (g) — the image that contains the final result *(the only trainable variable)*

## **Flow**
![Flow Chart](Images/flow.jpg)

## **Importing Libraries:**
Here I have implemeted the task using **Pytorch API** and other Image processing libraries such as **PIL (Pillow)** and mathematical library such as **Numpy**

## **Downloading a Pre-trained VGG19 model**
Here I am using the pre-trained VGG19 model. The model is trained on ImageNet images. And freezing the parameters.

## **UTILS:**
I recommand you to use a Cuda diver for faster computation and if you dont have a local GPU access you can use Google Colab's free GPU

### **Image Loader**
`load_image():`


> This functions have parameters:
>*   img_path : the local or relative path of the image to load
>*   max_size : Setting the maximum size of the image to 400px for computatonal reasons
>*  shape : the shape of the image
>
> This Function import the image from the given path and apply the necessary transformations to the image which includes resizing the image, converting the pixel values to Tensor and Normalizing the image
>
> This Function returns the Image tensor
---


### **Image converter**
`im_convert()`:
> This functions have parameters:
>*   tensor : The image tensor
>
> This function is used to convert back the normalized tensor to an numpy array
>
> The Function returns the Unormalized image array
---

### **Get Feature**
`get_features():`
> This functions have parameters:
>*   image : The image tensor
>*   model : The neural net
>*   layers: The name of the specfic layer of the neural Net
>
> This function takes inspiration from the Paper which i have read above, in this paper we are sampling the specific layers of the neural net for our task specially the first layers of each blocks of the VGG19, namely: `‘conv1_1’` , `‘conv2_1’` , `‘conv3_1’` , `‘conv4_1’` ,  `‘conv5_1’` and `‘conv5_2’`
>
> And then pass the image through each layer append it in a dict names `features`
>
> The Function returns the dict `features`
---
### Note
Intermediate Layers for Style and Content

Deeper layers of VGG-19 will extract the best and most complex features. Hence, conv5_2 is assigned to extract content components. From each block, the first convolution layers (shallow layers) i.e. from conv1_1 to conv5_1 detects multiple features like lines or edges. Refer to the image below.

![VGG Net](vgg.png)

The layers Highlighted in the Yellow are used for Style Feature Extraction and the layer which is Highlighted in Blue is used for content Feature Extraction

**One Important Note**

Here I am using the last layer of the last Convolution block for the computation of the loss fucntion whereas in the paper author have used every first layer of each block for the computation of the loss function

### Gram Matrix calculation
`gram_matrix()`
> This functions have parameters:
>*   tensor : the tensor of which we want to compute the Gram matrix
>
> The Function returns the Gram matrix of the input tensor


## Hyperparams 
Here we are setting the number of iteration as `steps=7000`
*   `α = 0.01` --> Content weight
*   `β = 10^7` --> Style Weight
*   `lr = 0.001` --> Learning Rate

We can also set up the individual weights for the feature layer for the loss computation, in the paper they have set the each weight as equal to one divided by the number of active layers with a non-zero loss-weight that is 0.2 in this case


## **Traing LOOP**

### **LOSS CALCULATION**
Let's talk about the loss calculation first!!

The final Loss function can be divided as the sum of two loss functions, namely `Content Loss` and `Style Loss`

#### **Content Loss**:
>* The content loss function ensures that the activations of the higher layers are similar between the content image and the generated image.
>
> As Discussed it the paper:
>
>* It is intuitive that higher layers of the model focus more on the features present in the image i.e. overall content of the image.
>* Content loss is calculated by Euclidean distance between the respective intermediate higher-level feature representation of input_image(x) (F) and content_image(p) (P) at layer l.
>
>    ![Content_loss](Images/Content_loss.png)
>
>* It is natural for a model to produce different feature maps in higher layers being activated in the presence of different objects.
>
>This helps us to deduce that images having the same content should also have similar activations in the higher layers.

#### **Style Loss**:
>* The style loss function makes sure that the correlation of activations in all the layers are similar between the style image and the generated image.
>* Style loss is conceptually different from Content loss.
>* We cannot just compare the intermediate features of the two images and get the style loss. That is why we have computed the GRAM matrix of the feature map.
>* Gram matrix is a way to interpret style information in an image as it shows the overall distribution of features in a given layer. It is measured as the amount of correlation present between features maps in a given layer.
>* Style loss is calculated by the distance between the gram matrices (or, in other terms, style representation) of the generated image (G) and the style reference image(E).
>
>   ![Style_L1](Style_L1.png)
>
>   ![Alt text](Images/styleL2.png)
>
>where the contribution of each layer in the style loss is depicted by some factor wl.


### **What's Happening in the training Loop??**
`train()`
> This functions have parameters:
>*   style_weights : the individual weight for the loss computation


we are computing the Content and Style loss and using the Adam Optimizer for backprop for `steps` number of iterations and printing loss and showing the target image every `show`th step

you can uncomment the 16th and 19th line and also commenting the 6th and 18th line for the exact implemetation of the paper
# convolutional-neural-network-visualization
In this project, I demonstrated how to visualize convolutional neural network (ConvNet) with tensorflow 2.x. Visualizing the intermediate steps taking place inside any ConvNet is critical for understanding how it works and improving existing model. This project includes four parts: 
1. visualize activation
2. visualize reconstruction from each layer
3. visualize classes 
4. visualize nueral styler tansfer

except that Visualizing activation and feature maps which is a forward propagation, visualzing reconstruction from a layer, visualizing classes, and visualizing neural style transfer are a backward propagation utilizing gradient descent. The model used in this project is VGG16, but other models would work similarly.

### visualize activation
The model is VGG16, a pretrained model with imagenet. Here I pass a image of dog into the model and record the activation of the image as it goes through each layer of the model.From the activation shown below, one can see that what the kernal does is detecting edges, circle and then gradually combine them to detect eyes, ear, noises and other shapes. 
![Capture1](https://user-images.githubusercontent.com/58440102/100679112-9f097080-3323-11eb-92d9-629e7cf656f3.PNG)
![Capture2](https://user-images.githubusercontent.com/58440102/100679106-9add5300-3323-11eb-9d82-8c056bf419ef.PNG)
![Capture3](https://user-images.githubusercontent.com/58440102/100679109-9ca71680-3323-11eb-89a7-a0af29820aa8.PNG)
![Capture4](https://user-images.githubusercontent.com/58440102/100679110-9dd84380-3323-11eb-894e-82cb82cee261.PNG)
![Capture5](https://user-images.githubusercontent.com/58440102/100679111-9e70da00-3323-11eb-9cbc-5a0fe602e015.PNG)

### visualize reconstruction from a layer
Reconstructing an image from its feature map is a direct application of gradient descent. Given the output from a layer for an input image, we can reconstruct it by initializing a random noise image and minimizing the loss (a content based loss function) with tf.GradientTap
![block1_conv1](https://user-images.githubusercontent.com/58440102/100680062-a2056080-3325-11eb-8280-befb290d9800.png)
![block1_conv2](https://user-images.githubusercontent.com/58440102/100680065-a3368d80-3325-11eb-9559-8a12f25a1c1c.png)
![block2_conv1](https://user-images.githubusercontent.com/58440102/100680070-a467ba80-3325-11eb-9c63-9b7de25316b8.png)

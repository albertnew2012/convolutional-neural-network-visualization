
# convolutional-neural-network-visualization
In this project, I demonstrated how to visualize a convolutional neural network (ConvNet) with TensorFlow 2.x. Visualizing the intermediate steps taking place inside any ConvNet is critical for understanding how it works and improving the existing model. This project includes four parts: 
1. visualize feature map and activation
2. visualize reconstruction from each layer
3. visualize classes 
4. visualize neural styler transfer

Except that Visualizing activation and feature maps which is a forward propagation, visualizing reconstruction from a layer, visualizing classes, and visualizing neural style transfer are backward propagation utilizing gradient descent. The model used in this project is VGG16, but other models would work similarly. The input image is a picture of a dog: 

![dog](https://user-images.githubusercontent.com/58440102/100690104-b30c9c80-333a-11eb-88d5-20123f463885.jpeg)

### visualize activation
The model is VGG16, a pre-trained model with imagenet. Here I pass an image of a dog into the model and record the activation of the image as it goes through each layer of the model. From the activation shown below, one can see that what the kernel does is detecting edges, circles,s and then gradually combine them to detect eyes, ears, noses, and other shapes. 

![Capture1](https://user-images.githubusercontent.com/58440102/100679112-9f097080-3323-11eb-92d9-629e7cf656f3.PNG)
![Capture2](https://user-images.githubusercontent.com/58440102/100679106-9add5300-3323-11eb-9d82-8c056bf419ef.PNG)
![Capture3](https://user-images.githubusercontent.com/58440102/100679109-9ca71680-3323-11eb-89a7-a0af29820aa8.PNG)
![Capture4](https://user-images.githubusercontent.com/58440102/100679110-9dd84380-3323-11eb-894e-82cb82cee261.PNG)
![Capture5](https://user-images.githubusercontent.com/58440102/100679111-9e70da00-3323-11eb-9cbc-5a0fe602e015.PNG)

### visualize reconstruction from a layer
Reconstructing an image from its feature map is a direct application of gradient descent. Given the output from a layer for an input image, we can reconstruct it by initializing a random noise image and minimizing the loss (a content-based loss function) with tf.GradientTap

![block1_conv1](https://user-images.githubusercontent.com/58440102/100680062-a2056080-3325-11eb-8280-befb290d9800.png)
![block1_conv2](https://user-images.githubusercontent.com/58440102/100680065-a3368d80-3325-11eb-9559-8a12f25a1c1c.png)
![block2_conv1](https://user-images.githubusercontent.com/58440102/100680070-a467ba80-3325-11eb-9c63-9b7de25316b8.png)

### visualize classes
Visualizing classes of the CNN model gives us more in deep knowledge of the model as to what the model is looking for in each class. visualizing classes is similar to visualizing reconstruction from a layer, except that the layer is the prediction layer. Each neuron among the 1000 neurons is detecting a specific class. Here I choose to visual 'French_bulldog', 'bubble', 'Persian_cat', 'violin', 'teddy', 'mountain_bike', 'mushroom','umbrella'. 

![ezgif com-gif-maker (1)](https://user-images.githubusercontent.com/58440102/100835065-d8280a80-3421-11eb-854a-c7520835906a.gif)

### visualize nueral styler tansfer
Neural style transfer can make a mundane, boring image artistic immediately. The process is very similar to other kinds of visualization as well. It starts by passing in the target image through a CNN model, the model will generate feature maps at each layer. These feature maps contain the style information of the target image. Then it takes another image to be transferred as initial input. Then using gradient descent to minimize the feature maps at each layer between the target image and initial image. The difference from reconstruction from a layer is the loss function. Here a style-based loss function is defined with a style matrix (Gram matrix).

![block1_conv1](https://user-images.githubusercontent.com/58440102/100680830-4f2ca880-3327-11eb-9522-de63cf29df89.png)
![block1_conv2](https://user-images.githubusercontent.com/58440102/100680831-4fc53f00-3327-11eb-9f61-10d7c51ca405.png)
![block2_conv1](https://user-images.githubusercontent.com/58440102/100680832-505dd580-3327-11eb-88c4-f34f39449195.png)
![block2_conv2](https://user-images.githubusercontent.com/58440102/100680833-50f66c00-3327-11eb-9f83-53d028d80c08.png)
![block3_conv1](https://user-images.githubusercontent.com/58440102/100680837-518f0280-3327-11eb-98be-b519849ddb0c.png)
![block3_conv2](https://user-images.githubusercontent.com/58440102/100680829-4e941200-3327-11eb-9943-d2b42a95c871.png)

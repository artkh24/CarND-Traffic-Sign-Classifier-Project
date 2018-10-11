# **Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./images/Random_images.jpg "Random six images"
[image6]: ./images/train_counts.jpg "Train dataset distribution"
[image7]: ./images/valid_counts.jpg "Validation dataset distribution"
[image8]: ./images/testing_counts.jpg "Test dataset distribution"
[image9]: ./German%20traffic%20signs/test1.jpg "Speed limit (50km/h)"
[image10]: ./German%20traffic%20signs/test2.jpg "Keep left"
[image11]: ./German%20traffic%20signs/test3.jpg "No entry"
[image12]: ./German%20traffic%20signs/test4.jpg "Roundabout mandatory"
[image13]: ./German%20traffic%20signs/test5.jpg "Stop"
[image14]: ./German%20traffic%20signs/test6.jpg "Slippery road"
[image15]: ./images/web_images_softmax.jpg "Softmax distribution"


---


### Data Set Summary & Exploration

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Here an example of six random images from dataset


![alt text][image5]
Below you can see images showing dataset distribution per class for train, validation and test 

![alt text][image6]
![alt text][image7]
![alt text][image8]

You can find example images for every class in Traffic_Sign_Classifier.html

### Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because it easier for the network to learn even when we are losing information about the color. And for MNIST dataset grayscaling gives good results

Then I normalize the data to the range(-1,1) because I learned that normalization helps in speed of training and performance and better convergence
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 	 			    |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 	 			    |
| Flattening			|flattens array, outputs 400					|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 1x1x400 	|
| Fully connected		| outputs 120        							|
| RELU					|												|
| Dropout			    |												|
| Fully connected		| outputs 84        							|
| RELU					|												|
| Dropout			    |												|
| Fully connected		| outputs 43        							|

 
To train the model, I used the Adam optimizer. And the final settings are
*batch size 128
*epochs 50
*learning rate 0.00088
*mu 0
*sigma 0.1
*dropout keep probability 0.5

My final model results were:
* validation set accuracy of 0.966 
* test set accuracy of 0.944

I started with pre-defined architecture LeNet I added two dropout layers that helps to increase the accuracy also played with hyperparameters to get good level of accuracy. I didn't go further to check other architectures like VGG or others but believe that they will increase the level of accuracy 
 

### Test a Model on New Images

Here are six German traffic signs that I found on the web:
And I see that there is no much difference between the images on web and the dataset images that can make the prediction difficult
these images are easily distinguishable than quite a few images from the original dataset, maybe the color range is slightly different

![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (50km/h)  | Speed limit (30km/h)   						| 
| Keep left     	    | Keep left 								    |
| No entry			    | No entry										|
| Roundabout mandatory	| Roundabout mandatory					 		|
|Stop                   | Stop                                          |
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.3%. 
Below there are softmax probabilities for each web downloaded image

![alt text][image15]


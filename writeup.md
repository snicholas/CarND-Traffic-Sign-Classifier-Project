# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
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
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/snicholas/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the vanilla python and numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data sets. On each row there are 4 images from train, validation and test data set respectively. 

![Example images](writeup_assets/exploratory.png)

And here it is a bar chart representing class distribution across the train data set.

![Example images](writeup_assets/exploratory2.png)


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

At first I tried using color RGB image normalized with the formula (image - 128)/128, but results were bad. Next I tried the same normaliztion tecnique with gray scale version of the images, with a good improvements. Lastly I tried converting RGB to YUV, with normalization  (image - dataset.mean)/dataset.std and this is the final version I decided to use. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x128 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x256  |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x256 				    |
| Fully connected		| 180 outputs       							|
| Fully connected		| 118 outputs       							|
| Fully connected		| 43 outputs        							|
| Softmax				| 1         									|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I optimized  reducing the mean of the error over the validation sets, maximizing the accuracy. i trained the model for 50 epochs with a batch size of 128 and learning rate of 0.001. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.
The architecture used is based on the LeNet developed during the lessons. I've changed the filter sizes and added an additional fully connected layer. This modifications, plus the preprocessing, augmented the final validation scores over the 0.93 accuracy required.

My final model results were:
* validation set accuracy of 0.956 
* test set accuracy of 0.946

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![](writeup_assets/01.jpg) ![](writeup_assets/02.jpg) ![](writeup_assets/03.jpg) ![](writeup_assets/04.jpg) ![](writeup_assets/05.jpg)

To be honest I'm not very sure why the first image is being missclassified, and watching to the probabilities the corrected prediction was second in place with 43% versus the 56% of the wrong one.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)  | Dangerous curve to the left       			| 
| Road work    			| Road work										|
| Keep left				| Keep left										|
| Priority road			| Priority road      							|
| Yield 	      		| Yield     					 				|



The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 94.6% .

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Dangerous curve to the left (probability of 0.58), but it was a Speed limit (30km/h). 
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .58         			| Stop sign   									| 
| .43     				| Speed limit (30km/h)							|
| .07					| Go straight or right							|
| .00	      			| Speed limit (20km/h)			 				|
| .00				    | Road work         							|


For the second image the prediction was good with an almost 100% certainity.  

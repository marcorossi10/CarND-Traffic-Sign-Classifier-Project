# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

Here is the link to my GitHub repository for this project: https://github.com/marcorossi10/CarND-Traffic-Sign-Classifier-Project

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

In this section (cell number 2) of the notebook I will obtain useful information starting from the imported data. I will calculate the size of the training set, of the validation set and of the testing set. Here, it is also shown the input images shape. Moreover, I define a function to get the number of the classes in the data set by reading the last row of the file `signnames.csv`. In this way the pipeline will work even if new sign will be added to the file.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

This part is addressed from cell number 3 to 5 of the notebook.
In this section I will plot one image of each type included in the training data set. In the title of the image I will also write the corresponding description; I achieved that by implementing the function `find_sign_description()` that returns the description of the signal based on the input label (reading directly from the file `signnames.csv`).

Note that the input data are shuffled to run the algorithm each time on a different set of images. In this way I could have a rough idea of the training set and if there were correspondence between the label and the image itself.

In the visualization part, when adjacents images have long titles, it might be that these titles collide.


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The preprocessing of the data takes place in the cell number 6 of the notebook. 
It consists only on the implementation of the function data_normalization(). This function is implementing the suggestion given in the project comments: its main aim is to obtain a distribution of the data that will resemble a Gaussian curve centered at zero.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The starting point of my work has been the LeNet architecture and the analysis of the result/accuracy on the validation set. 
The results were already "decent" since the final accuracy was roughly oscillating between 0.82 and 0.86.

To improve the above mentioned result I implemented the dropout technique at the end of each layer.
....
....   THIS POINTS MIGHT GO LATER IN THE NEXT SESSIONS
...

This approach increased the final accuracy on the validation set quite drastically, leading to a final value included (most of the times) between 0.95 and 0.94 (in all the test-run I tried the results were higher than the required threshold of 0.93)


My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|



       
For the second image ... 



EXAMPLE WHERE THERE WAS A FAILURE ON THE FIVE IMAGE TEST.. WRITE SOME COMMENTS ON THE WRONGLY CLASSIFIED IMAGE AND SAY THAT IT WAS HAPPENING APPROX 1 TIME EVERY 20 DIFFERENT MODEL TRAININGS

The error I encountered was mainly a wrong classification of the speed limit sign (generally confusing 20 km/h with 30 or 50 Km/h)


INFO:tensorflow:Restoring parameters from ./sign_recognition
TopKV2(values=array([[  6.120e-01,   3.732e-01,   5.896e-03,   5.357e-03,   1.821e-03],
       [  8.742e-01,   1.258e-01,   1.798e-05,   4.896e-06,   3.788e-07],
       [  9.998e-01,   1.189e-04,   4.936e-05,   5.735e-06,   1.968e-07],
       [  1.000e+00,   9.568e-12,   5.287e-12,   4.942e-14,   4.049e-14],
       [  9.999e-01,   6.769e-05,   7.937e-07,   5.344e-07,   4.779e-07]], dtype=float32), indices=array([[ 0,  1, 14,  2,  4],
       [17, 14,  1,  0, 29],
       [ 9, 10, 41, 12, 13],
       [12, 32, 13, 42, 17],
       [14,  1,  5, 29, 17]], dtype=int32))

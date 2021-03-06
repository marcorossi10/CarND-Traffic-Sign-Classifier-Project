# **Traffic Sign Recognition** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

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

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolutional		  	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| DROPOUT				| 0.85 of keep probability						|
| Max pooling	      	| 2x2 stride, VALID padding,  outputs 14x14x6 	|
| DROPOUT				| 0.85 of keep probability						|
| Convolutional   	    | 1x1 stride, VALID padding, outputs 10x10x16	|
| RELU					|												|
| DROPOUT				| 0.85 of keep probability						|
| Max pooling	      	| 2x2 stride, VALID padding,  outputs 5x5x16 	|
| DROPOUT				| 0.85 of keep probability						|
| Flattening   			| outputs 400  									|
| Fully connected		| outputs 120 									|
| RELU					|												|
| DROPOUT				| 0.85 of keep probability						|
| Fully connected		| outputs 84 									|
| RELU					|												|
| DROPOUT				| 0.85 of keep probability						|
| Fully connected		| outputs 43 									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a batch size of 128 and 25 epochs. Here I decided to increase (with rispect to the LeNet implementation) the number of epochs in order to increase the forward passes and backward passes of all the training examples through the network. This generally leads to a better training of the network but it is also more time-consuming.

The learning rate and the type of the optimizer (AdamOptimizer) have not been modified since they were suggested in the lessons as good default values (and indeed they have been so).

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The starting point of my work has been the LeNet architecture and the analysis of the result/accuracy on the validation set. 
The first step I had to compute has been to adapt the input and the output layer respectively to the new image input sizes (RGB images so 32x32x3) and to the number of classes (43) representing the traffic signes.
After these two first modifications I run already the network to check the results: they were already "decent" since the final accuracy (on the validation set) was roughly oscillating between 0.82 and 0.86. In any case, I was facing an under-fitting scenario and I was still too far away from the required accuracy. 

To improve the above mentioned result I implemented the dropout technique at the end of each layer.
This technique randomly drops neurons of the network and ignore them during training (ONLY during training). Interesting/intuitive explanation on dropout (https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/):

`As a neural network learns, neuron weights settle into their context within the network. Weights of neurons are tuned for specific features providing some specialization. Neighboring neurons become to rely on this specialization, which if taken too far can result in a fragile model too specialized to the training data. This reliant on context for a neuron during training is referred to complex co-adaptations.

You can imagine that if neurons are randomly dropped out of the network during training, that other neurons will have to step in and handle the representation required to make predictions for the missing neurons. This is believed to result in multiple independent internal representations being learned by the network.`

This approach increased the final accuracy on the validation set quite drastically, leading to a set of  final accuracies that were included between 0.955 and 0.935 for each new trained model.

An example of final model results are:
* training set accuracy of 0.998
* validation set accuracy of 0.945
* test set accuracy of 0.949


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

In the cell number 14 of the notebook I imported and plotted the five images downloaded from the web.

I chose on purpose a traffic sign with a speed limit (the file named 'five_images_example/30Km_h.jpg'), because I wanted to see if and how the network was correctly classifying the maximum allowed speed in the sign.
Moreover I also chose the priority road sign (the file named 'five_images_example/Priority.jpg') since it does not contain "strong" colors such red and blue.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

I run the entire algorithm (including the training part) multiple times. I did this since the data are randomly shuffled each time and I wanted to evaluate if the decided architecture and the selected hyperparameters were always performing well on the training/validation/test sets. 

The model was able to correctly guess 5 of the 5 traffic signs most of the times. 
However, in some cases (roughly speaking 1 time every 20), the speed limit sign was not correctly classified: the details will be given in the next paragraph.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 15th and 16th cells of the Ipython notebook.

First I will show the results for a model that classify all the test images correclty.

**30 KM/H SIGN**

| Probability         	|     Prediction	     	| 
|:---------------------:|:-------------------------:| 
| 9.29e-01    			| Speed limit (30km/h) 		| 
| 6.63e-02				| Speed limit (20km/h)		|
| 2.44e-03				| Speed limit (50km/h)		|
| 1.42e-03    			| Children crossing			|
| 9.35e-04			    | Wild animals crossing		|

**NO ENTRY SIGN**

| Probability         	|     Prediction	     	| 
|:---------------------:|:-------------------------:| 
| 1.00         			| No entry   				| 
| 2.07e-09    			| Stop 						|
| 8.69e-11				| No passing				|
| 6.14e-14	     		| Speed limit (70km/h)		|
| 5.74e-14			    | Speed limit (30km/h)  	|

**NO PASSING SIGN**

| Probability         	|     Prediction	        				| 
|:---------------------:|:-----------------------------------------:| 
| 1.00           		| No passing   								| 
| 6.01e-08     			| End of no passing 						|
| 6.07e-09				| Vehicles over 3.5 metric tons prohibited	|
| 1.26e-09	      		| No entry					 				|
| 4.50e-10			    | Priority road    							|

**ROAD WITH PRIORITY SIGN**

| Probability         	|     Prediction	        				| 
|:---------------------:|:-----------------------------------------:| 
| 1.00           		| Priority road  							| 
| 1.12e-12    			| Right-of-way at the next intersection		|
| 7.45e-14				| Traffic signals							|
| 2.17e-14	      		| No entry					 				|
| 1.06e-14			    | Road work      							|

**STOP SIGN**

| Probability         	|     Prediction	      	| 
|:---------------------:|:-------------------------:| 
| 1.00          		| Stop sign   				| 
| 3.15e-05     			| Speed limit (30km/h) 		|
| 2.43e-05				| Slippery road				|
| 8.50e-06	      		| Speed limit (50km/h)		|
|  6.00e-06			    | Speed limit (60km/h)  	|

As it is possible to be seen from the tables the model is quite sure on what it is classifying, since there is always at least one order of magnitude of difference between the first and the second probability.

On the other hand, as mentioned in the previous section, sometime models that were misclassifying the speed limit sign were found. Generally the "wrong" models were confusing 20 km/h with 30 or 50 Km/h.
In the next final table I report the probabilities obtained in such a situation for the speed limit sign:

**30 KM/H SIGN**

| Probability         	|     Prediction	        | 
|:---------------------:|:-------------------------:| 
| 6.12e-01    			| Speed limit (20km/h) 		| 
| 3.73e-01				| Speed limit (30km/h) 		|
| 5.89e-03				| Stop						|
| 5.35e-03    			| Speed limit (50km/h)		|
| 1.82e-03			    | Speed limit (70km/h)     	|

Is it possible to see how this particular model is uncertain between different speed limits values, especially between 20 and 30 Km/h.


#### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

#### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```
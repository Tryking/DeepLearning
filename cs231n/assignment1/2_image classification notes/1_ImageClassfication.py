"""
http://cs231n.github.io/classification/
"""
"""
Challenges. 
Since this task of recognizing a visual concept (e.g. cat) is relatively trivial for a human to perform, 
it is worth considering the challenges involved from the perspective of a Computer Vision algorithm. 
As we present (an inexhaustive) list of challenges below, keep in mind the raw representation of images as a 3-D array of brightness values:

Viewpoint variation. 
A single instance of an object can be oriented in many ways with respect to the camera.
Scale variation. 
Visual classes often exhibit variation in their size (size in the real world, not only in terms of their extent in the image).
Deformation. 
Many objects of interest are not rigid bodies and can be deformed in extreme ways.
Occlusion. 
The objects of interest can be occluded. Sometimes only a small portion of an object (as little as few pixels) could be visible.
Illumination conditions. The effects of illumination are drastic on the pixel level.
Background clutter. 
The objects of interest may blend into their environment, making them hard to identify.
Intra-class variation. The classes of interest can often be relatively broad, such as chair. There are many different types of these objects, each with their own appearance.

A good image classification model must be invariant to the cross product of all these variations, while simultaneously retaining sensitivity to the inter-class variations.

Data-driven approach. How might we go about writing an algorithm that can classify images into distinct categories? 
Unlike writing an algorithm for, for example, sorting a list of numbers, 
it is not obvious how one might write an algorithm for identifying cats in images. 
Therefore, instead of trying to specify what every one of the categories of interest look like directly in code, 
the approach that we will take is not unlike one you would take with a child: 
we’re going to provide the computer with many examples of each class and then develop learning algorithms that look at these examples 
and learn about the visual appearance of each class. 
This approach is referred to as a data-driven approach, since it relies on first accumulating a training dataset of labeled images. 

The image classification pipeline. 
We’ve seen that the task in Image Classification is to take an array of pixels that represents a single image and assign a label to it. 
Our complete pipeline can be formalized as follows:

Input: 
        Our input consists of a set of N images, each labeled with one of K different classes. We refer to this data as the training set.
Learning: 
        Our task is to use the training set to learn what every one of the classes looks like. 
            We refer to this step as training a classifier, or learning a model.
Evaluation: 
        In the end, we evaluate the quality of the classifier by asking it to predict labels for a new set of images that it has never seen before. 
        We will then compare the true labels of these images to the ones predicted by the classifier. 
        Intuitively, we’re hoping that a lot of the predictions match up with the true answers (which we call the ground truth).
        
        
Nearest Neighbor Classifier

As our first approach, we will develop what we call a Nearest Neighbor Classifier. 
This classifier has nothing to do with Convolutional Neural Networks and it is very rarely used in practice, 
but it will allow us to get an idea about the basic approach to an image classification problem.

Example image classification dataset: CIFAR-10. One popular toy image classification dataset is the CIFAR-10 dataset. 
This dataset consists of 60,000 tiny images that are 32 pixels high and wide. 
Each image is labeled with one of 10 classes (for example “airplane, automobile, bird, etc”). 
These 60,000 images are partitioned into a training set of 50,000 images and a test set of 10,000 images. 
In the image below you can see 10 random example images from each one of the 10 classes:


*************************************
Suppose now that we are given the CIFAR-10 training set of 50,000 images (5,000 images for every one of the labels), 
and we wish to label the remaining 10,000. 
The nearest neighbor classifier will take a test image, compare it to every single one of the training images, 
and predict the label of the closest training image. 
In the image above and on the right you can see an example result of such a procedure for 10 example test images. 
Notice that in only about 3 out of 10 examples an image of the same class is retrieved, while in the other 7 examples this is not the case. 
For example, in the 8th row the nearest training image to the horse head is a red car, presumably due to the strong black background. 
As a result, this image of a horse would in this case be mislabeled as a car.

You may have noticed that we left unspecified the details of exactly how we compare two images, which in this case are just two blocks of 32 x 32 x 3. 
One of the simplest possibilities is to compare the images pixel by pixel and add up all the differences. 
In other words, given two images and representing them as vectors I1,I2 , a reasonable choice for comparing them might be the L1 distance:

d1(I1,I2)=∑p|Ip1−Ip2|

Where the sum is taken over all pixels. 

Let's also look at how we might implement the classifier in code. 
First, let's load the CIFAR-10 data into memory as 4 arrays: the training data/labels and the test data/labels. 
In the code below, Xtr (of size 50,000 x 32 x 32 x 3) holds all the images in the training set, 
and a corresponding 1-dimensional array Ytr (of length 50,000) holds the training labels (from 0 to 9):

******************************************************************************
Summary
In summary:

We introduced the problem of Image Classification, in which we are given a set of images that are all labeled with a single category. 
We are then asked to predict these categories for a novel set of test images and measure the accuracy of the predictions.
We introduced a simple classifier called the Nearest Neighbor classifier. 
We saw that there are multiple hyper-parameters (such as value of k, or the type of distance used to compare examples) 
that are associated with this classifier and that there was no obvious way of choosing them.
We saw that the correct way to set these hyperparameters is to split your training data into two: a training set and a fake test set, 
which we call validation set. We try different hyperparameter values and keep the values that lead to the best performance on the validation set.
If the lack of training data is a concern, we discussed a procedure called cross-validation, which can help reduce noise in estimating which hyperparameters work best.
Once the best hyperparameters are found, we fix them and perform a single evaluation on the actual test set.
We saw that Nearest Neighbor can get us about 40% accuracy on CIFAR-10. 
It is simple to implement but requires us to store the entire training set and it is expensive to evaluate on a test image.
Finally, we saw that the use of L1 or L2 distances on raw pixel values is not adequate since the distances correlate more strongly with backgrounds 
and color distributions of images than with their semantic content.
In next lectures we will embark on addressing these challenges and eventually arrive at solutions that give 90% accuracies, 
allow us to completely discard the training set once learning is complete, and they will allow us to evaluate a test image in less than a millisecond.


Summary: Applying kNN in practice
If you wish to apply kNN in practice (hopefully not on images, or perhaps as only a baseline) proceed as follows:

Preprocess your data: Normalize the features in your data (e.g. one pixel in images) to have zero mean and unit variance. 
We will cover this in more detail in later sections, and chose not to cover data normalization in this section because pixels in images 
are usually homogeneous and do not exhibit widely different distributions, alleviating the need for data normalization.
If your data is very high-dimensional, consider using a dimensionality reduction technique such as PCA (wiki ref, CS229ref, blog ref) or even Random Projections.
Split your training data randomly into train/val splits. 
As a rule of thumb, between 70-90% of your data usually goes to the train split. 
This setting depends on how many hyperparameters you have and how much of an influence you expect them to have. 
If there are many hyperparameters to estimate, you should err on the side of having larger validation set to estimate them effectively. 
If you are concerned about the size of your validation data, it is best to split the training data into folds and perform cross-validation. 
If you can afford the computational budget it is always safer to go with cross-validation (the more folds the better, but more expensive).
Train and evaluate the kNN classifier on the validation data (for all folds, if doing cross-validation) for many choices of k (e.g. the more the better) 
and across different distance types (L1 and L2 are good candidates)
If your kNN classifier is running too long, consider using an Approximate Nearest Neighbor library (e.g. FLANN) to accelerate the retrieval (at cost of some accuracy).
Take note of the hyperparameters that gave the best results. 
There is a question of whether you should use the full training set with the best hyperparameters, since the optimal hyperparameters might change if you were to fold the validation data into your training set (since the size of the data would be larger). In practice it is cleaner to not use the validation data in the final classifier and consider it to be burned on estimating the hyperparameters. Evaluate the best model on the test set. Report the test set accuracy and declare the result to be the performance of the kNN classifier on your data.
"""


















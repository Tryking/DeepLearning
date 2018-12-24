"""
In the last section we introduced the problem of Image Classification,
which is the task of assigning a single label to an image from a fixed set of categories.
Morever, we described the k-Nearest Neighbor (kNN) classifier which labels images by comparing them to (annotated) images from the training set.
As we saw, kNN has a number of disadvantages:

    The classifier must remember all of the training data and store it for future comparisons with the test data.
        This is space inefficient because datasets may easily be gigabytes in size.
    Classifying a test image is expensive since it requires a comparison to all training images.
Overview. We are now going to develop a more powerful approach to image classification that
    we will eventually naturally extend to entire Neural Networks and Convolutional Neural Networks.
The approach will have two major components: a score function that maps the raw data to class scores,
and a loss function that quantifies the agreement between the predicted scores and the ground truth labels.
We will then cast this as an optimization problem in which we will minimize the loss function with respect to the parameters of the score function.

"""
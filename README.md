Project-Street-View-House-Number-Recognition
==============================================
Recognizing digits from natural scene images using CNN Model with momentum

Dataset
-------
Street View House Number (SVNH) dataset has been used for this project. This contains 73257 samples of 32x32 images. I am using the format 2 of this dataset where each image is centered around a single character. Dataset has been downloaded from this link : http://ufldl.stanford.edu/housenumbers/

### Prediction ###
------
Output will be 0 - 9 digit.

### Implementation ###
------
I used a CNN with 2 convolution-pooling layers and a fully connected ANN unit with 1 hidden layer of size 500 activation units at the end, along with momentum. Each convolution filter is of size 5x5 and pool size used for downsampling is 2x2. First convolution filter is having number of feature maps out as 20 and second one has 50.

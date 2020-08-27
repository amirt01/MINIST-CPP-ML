# MNIST-CPP-ML
Application of KNN and KMean algorithm applied to the MNIST database of hand written digits.

## About the Data
TL;DR: The MNIST database of handwritten digits has two sets of data: training and testing. Each database comes with a set of images and a set of labels. The training data set has 60,000 examples, and the test set has 10,000 examples.

All information about the MNIST database can be found at this link: http://yann.lecun.com/exdb/mnist/

### Important Details:
The MNIST database is formatted such that each file has a header, and then is followed by their respective set of images stored in unsigned bytes. In this project, I used this header to validate the file and then store all of the training data in an std::vector. I split up this vector into three separate std::vectors for training, testing, and validating my model. I did not use the test database for this project, but this can easily be added later.

## The Algorithms
I implemented two algorithms for this project, K-Nearest Neighbors (KNN) and K-Means Clustering. For this project, each algorithm is represented by a class that both inherit from a common algorithm virtual base class to make my code cleaner and easier to read.

## What I Learned
While completing this project, I learned a lot about working with large databases and the beginnings of Machine Learning in C++. I learned about the supervised learning and working with labels. I also learned about multi-class classification with the nine different digits. I was also introduced to the idea of clustering through K-Means Clustering.

## What Next
I plan to improve upon this project by including more algorithms as I learn them and a graphical interface to make this project more accessible to a broader audience.

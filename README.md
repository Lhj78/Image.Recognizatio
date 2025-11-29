This program builds an image classification model using a Convolutional Neural Network (CNN) on the CIFAR-10 dataset. CIFAR-10 contains 60,000 small color images of size 32x32 pixels, belonging to 10 different categories such as airplane, car, bird, cat, deer, dog, frog, horse, ship, and truck. The dataset is automatically downloaded using TensorFlow.

First, the images are normalized by dividing pixel values by 255 so they fall between 0 and 1. This helps the network train faster. The labels are flattened so they become a simple 1-dimensional array of class numbers instead of a column vector.

Next, class names are created to display human-readable labels later.

A CNN model is then built. The first layer is a Conv2D layer with 32 filters, each filter being a 3×3 kernel. This layer detects basic image features like edges and patterns. After that a MaxPooling layer is used, which reduces the size of the feature maps to make computation faster. This process repeats: a second Conv2D layer with 64 filters learns more detailed features, followed by another MaxPooling layer.

The output of the convolutional layers is then flattened into a 1-dimensional vector. After flattening, a Dense (fully-connected) layer with 128 neurons learns higher-level patterns. A Dropout layer with 30% dropout is used to reduce overfitting by randomly disabling some neurons during training. Finally, a Dense layer with 10 neurons and softmax activation gives probabilities for each of the 10 classes.

The model is compiled using the Adam optimizer (which adjusts the learning rate automatically), the sparse categorical cross-entropy loss function (used when labels are integers from 0 to 9), and accuracy as the evaluation metric.

The model is trained for 10 epochs using 90% of the training data, while 10% is used for validation. After training, the model is evaluated on the test set of 10,000 images to measure accuracy on new, unseen data.

Predictions are then made for the test images. For the first five test images, the program displays the image and prints both the predicted label and the actual label. Softmax gives a probability for each class, and the class with the highest probability is selected as the prediction.

Overall, the program demonstrates how to load data, preprocess images, build a CNN, train it, evaluate its accuracy, and finally make predictions and visualize results. It is a complete pipeline for image classification using deep learning.

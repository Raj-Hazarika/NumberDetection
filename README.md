# Number Detection

This project uses neural network to detect the number drawn by the user in a pygame window. The neural network model is developed using tensorflow and sci-kit learn libraries. The project also has a GUI which lets the user draw a number, this GUI was developed using pygames.

##  Detailed Explanation

- The project will involve creating a Pygame window where the user can draw a number using their mouse. Once the user is satisfied with their drawing, they can press the 's' letter, this will save the image and activate the neural network, which will then analyze the drawing and make a prediction of the number that was drawn. 

- Before executing the machine learning model the saved image will be converted to a 28x28 dimension image and the rgb values will also be converted to grayscale. This is done so that the machine learning model doesn't have to consider those factor and it lowers the load on the machine.

- The neural network will be trained on a dataset of images of handwritten numbers, so that it can learn to recognize and differentiate different numbers based on their visual characteristics. In the current project, the neural network is already trained and save as "NumModel.h5" file. In this model, I used 7 different layers which includes 2 Conv2D layers, 2 MaxPool layers, 2 BatchNormalization layers and one GlobalAveragePooling layer. Currently, the trained model has an accuracy of over 95%.

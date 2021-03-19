# Static_Facial_Expression_Recognition

Contains code to use a VGG model trained AffectNet to recognize four primary emotional classes (happiness, sadness, anger, neutral).

The code was developed using Tensorflow, Keras and OpenCV.
1. Tensorflow 1.14.0
2. Keras 2.3.0
3. OpenCV 4.1.1
4. python 3.7.4

Downloading the model is required before running the code. Please download the model (https://utdallas.box.com/s/3b16v2yygj73hx31tjs5w1wqqbmlqt21) and place it in the output directory.

The demo is in a .py and jupyter notebook. Emo_Detect contains the demo. Simply modify the glob() function to point to the image files. When the program is done running a results.csv will be generated in the same directory.

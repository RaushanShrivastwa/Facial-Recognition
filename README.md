# Facial-Recognition

This project implements a facial recognition system that verifies whether a given image matches the face detected in a live camera feed. Built using TensorFlow, Keras, and OpenCV, the system leverages deep learning for accurate facial feature extraction and comparison.

🔍 Features:

Real-time face detection and recognition
Face verification by comparing an input image with camera feed
Custom deep learning model using Conv2D, BatchNormalization, and Dense layers
Uses cosine similarity or Euclidean distance for face embedding comparison

🛠️ Tech Stack:

Python
TensorFlow & Keras (for model design and training)
OpenCV (for video capture and face detection)
NumPy

🚀 How It Works:

Capture and process a reference face image.
Continuously scan frames from the webcam.
Extract facial features using a trained CNN model.
Compare the features to determine if the person in the frame matches the input image.

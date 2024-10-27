**Traffic Sign Detection Using CNN**

This project is a tool to detect and identify traffic signs from images or real time camera function, which can help make driving safer. By recognizing traffic signs, this technology can help driver assistance systems and be used in self-driving cars.

**Introduction** - Traffic Sign Detection is a project that uses a type of deep learning called Convolutional Neural Networks (CNNs) to find and classify traffic signs in images. It’s like giving a computer the ability to "see" road signs and understand what they mean, just like a human driver would.

**Problem Statement** -
With more cars on the road, traffic safety is a big concern. Recognizing traffic signs quickly and accurately is important to prevent accidents and guide drivers. Our project addresses this by training a model to detect signs from images, making it suitable for use in automated driving systems.

**Technologies Used** -

Python: Main programming language for building and running the project.

TensorFlow: A tool for machine learning that lets us create and train the CNN model.

OpenCV: Used for image processing, like resizing or enhancing images before feeding them to the model.

NumPy: A library for handling and processing large amounts of data.

Matplotlib: Helps in creating graphs and visualizing the data and results.


**How It Works** -

1.Data Collection: We use a dataset of traffic signs that contains different types of signs.

2.Data Preprocessing: We prepare the images by resizing, normalizing, and organizing them for training.

3.Model Training: The CNN model is trained to recognize patterns in the images so it can identify various traffic signs.

4.Testing and Evaluation: After training, the model is tested to see how accurately it can detect and classify signs it hasn’t seen before.


**Usage**

Train the Model: Run this command to start training -->
python train_model.py

Detect Traffic Signs in Real-Time: Use this command to run real-time sign detection --> 
python app.py

Note: Make sure your webcam is on for real-time detection.

**Interface**
![image](https://github.com/user-attachments/assets/4bf7489c-b5db-461f-a745-67670f73e957)
**UPLOAD IMAGE**
![image](https://github.com/user-attachments/assets/22e15fd5-66ea-481a-b2b9-9831b81b29fe)
![image](https://github.com/user-attachments/assets/20dd108e-e4fc-4ca0-be01-e97cad4a17fb)
**USE CAMERA**
![image](https://github.com/user-attachments/assets/ce469f87-8741-41e0-b150-320519654871)
![image](https://github.com/user-attachments/assets/b1de449e-40c6-45f2-97e3-f02a4b9b85e2)

**Results** - 
The model is trained to recognize various types of traffic signs with an accuracy of over 95%.
It can recognize multiple types of signs, such as stop signs, no parking, and narrow bridge signs etc.


**Future Scope** - 

1.Expand the Dataset: Include more traffic signs from different countries to make it usable globally.

2.Improve Speed: Make the detection faster for better real-time performance.

3.Enhanced Safety Features: This project can be integrated with more safety measures in self-driving cars and advanced driver-assistance systems.

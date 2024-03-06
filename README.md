# Yoga-pose-classification-using-YOLOv8
This project aims to classify yoga poses using the YOLOv8 model for keypoint detection and classification. The process involves downloading a yoga pose image dataset from Kaggle, detecting keypoint coordinates using YOLOv8m-pose.pt model, generating a dataset from the keypoints, training a neural network for pose classification, and creating a Streamlit web application for real-time pose classification.
## Dataset
The yoga pose image dataset was obtained from Kaggle and used for training and testing purposes. The dataset includes various yoga poses such as Downdog, Goddess, Plank, Tree, and Warrior2.
## Key Steps
1. Download Dataset: The yoga pose image dataset is downloaded from Kaggle using the opendatasets library.
2. Keypoint Detection: Utilize the YOLOv8m-pose.pt model to detect keypoint coordinates of each yoga pose image in the dataset.
3. Dataset Generation: Generate a dataset from the detected keypoints by running all the images through the YOLOv8 model.
4. Neural Network Training: Train a neural network for pose classification using Keras, Adam optimizer, and Sparse Categorical Cross-Entropy loss function. Achieved a train accuracy of 96% and test accuracy of 95%.
5. Streamlit Web Application: Develop a Streamlit web application for real-time pose classification. Users can upload an image, and the application will classify yoga poses such as Downdog, Goddess, Plank, Tree, and Warrior2.
## Requirements
- Ensure you have the following dependencies installed:
Python 3.11
OpenCV-Python
TensorFlow 2.15.0
Streamlit 1.30.0
Pydantic 2.6.1
Ultralytics 8.1.20
NumPy 1.26.3
You can install the dependencies using the provided requirements.txt file:
pip install -r requirements.txt

## License
This project is licensed under the [Your License Name] License - see the LICENSE.md file for details.

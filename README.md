# Teknofest--Artificial-Intelligence-in-Health

## Project Overview

This project is developed as part of the Teknofest initiative, focusing on the application of Artificial Intelligence in healthcare. The main goal is to analyze brain tomography images to detect and classify stroke conditions and identify damaged areas through segmentation.

### Objectives

1. **Stroke Detection**: Classify images to determine whether the patient has a stroke or not.
2. **Stroke Classification**: If a stroke is detected, classify the type as either ischemic or hemorrhagic.
3. **Segmentation**: Accurately segment the damaged area in the brain to localize the stroke.

### Project Goals

Our aim is to ensure:
- Accurate classification of brain images.
- Precise detection of damaged brain areas due to stroke.

## Technologies and Tools Used

- **IDE**: PyCharm
- **Medical Image Processing**: SimpleITK (to convert DICOM images to PNG format)
- **Image Cleaning**: OpenCV, NumPy (for pre-processing and cleaning the image data)
- **Model Training**: TensorFlow, Keras (for building and training the models)
- **Prediction and Evaluation**: Scikit-learn (for model evaluation and predictions)
- **Data Output**: Final results are saved in CSV format.

### Data

The dataset used consists of brain tomography images in **DICOM** format, which are processed for analysis.

### Machine Learning Models

1. **Convolutional Neural Networks (CNN)**: Used for image classification tasks to determine the presence of a stroke and classify its type.
2. **U-Net Model**: Applied for segmentation, identifying and localizing damaged areas in the brain.

## How to Run the Project

### Prerequisites

- Python 3.x
- PyCharm or any other Python-compatible IDE
- Required libraries:
  - `SimpleITK`
  - `OpenCV`
  - `NumPy`
  - `TensorFlow`
  - `Keras`
  - `Scikit-learn`
  - `Pandas` (for CSV file handling)

You can install the necessary libraries using the following command:

```bash
pip install -r requirements.txt

# Pneumonia and Brain Tumor Detection Web App

This web application is designed to detect pneumonia and brain tumors using Flask, HTML, CSS, and TensorFlow. It utilizes pre-trained deep learning models for predictions based on uploaded images.

## Overview

The application has two main functionalities:

1. **Pneumonia Detection**
   - Allows users to upload chest X-ray images for pneumonia detection.
   - Uses a pre-trained model (`best_model.h5`) to predict pneumonia presence or absence.

2. **Brain Tumor Detection**
   - Provides a feature to upload brain MRI images for brain tumor detection.
   - Utilizes a pre-trained model (`b11_model.h5`) to predict the presence of a brain tumor.

## Setup and Usage

Follow these steps to set up and run the application locally:

1. **Installation**
   - Ensure Python is installed on your system (Python 3.x recommended).
   - Install the required packages by running `pip install -r requirements.txt`.

2. **Models**
   - Place the pre-trained models (`best_model.h5` and `b11_model.h5`) in the root directory.

3. **Running the App**
   - Start the Flask server by running `python app.py`.
   - Access the application through a web browser at `http://localhost:5000`.

## Structure

- `app.py`: Contains the Flask application setup and routes for image uploads and predictions.
- `index.html`: Homepage template.
- `pneumonia.html`: Template for pneumonia detection.
- `brain_tumor.html`: Template for brain tumor detection.
- `requirements.txt`: Includes all necessary Python packages and their versions.

## Note

- **Development Server Warning:** This app uses a development server. For production use, deploy the app using a production-ready WSGI server.
- Ensure proper error handling and security measures, especially when dealing with file uploads.

## Contribution

Contributions are welcome! Feel free to open issues or submit pull requests for improvements or bug fixes.



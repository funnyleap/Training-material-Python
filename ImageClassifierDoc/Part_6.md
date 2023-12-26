# Technical Documentation for Celebrity Recognition Model Deployment and Prediction Script

## Overview
This script details the process of downloading a trained machine learning model from an AWS S3 bucket, preprocessing images stored in S3, and using the model to predict whether an image is of a specific celebrity. The script integrates AWS services, TensorFlow, Keras, and image processing techniques.

## Libraries and Modules
- `numpy` (imported as `np`): A library for numerical and array operations in Python.
- `io.BytesIO`: An in-memory bytes buffer used for binary I/O operations.
- `tensorflow.keras.preprocessing.image`: A module for image preprocessing utilities in Keras.
- `tensorflow.keras.models.load_model`: A function to load a saved Keras model.
- `boto3`: AWS SDK for Python, used for interfacing with AWS services, particularly S3.

## Script Components

### Model Download from AWS S3
```python
bucket_name = 'celebs3bucket'
model_file_key = 'models/image_recognition_model.h5'
s3 = boto3.client('s3')
local_model_file = 'downloaded_model.h5'
s3.download_file(bucket_name, model_file_key, local_model_file)
loaded_model = load_model(local_model_file)
```
- **AWS S3 Client Initialization**: Establishes a connection to AWS S3.
- **Model File Download**: Downloads the saved model file from S3 to a local file path.
- **Model Loading**: Loads the downloaded model file into `loaded_model` for further use.

### Function: `preprocess_image_from_s3`
#### Purpose
Fetches and preprocesses an image from an S3 bucket for model input.

#### Parameters
- `bucket_name`: The name of the S3 bucket.
- `key`: The key (path) of the image in the S3 bucket.
- `target_size`: The desired size of the image (default is 224x224 pixels).

#### Process
1. Fetches the image data from S3.
2. Loads the image using `tensorflow.keras.preprocessing.image.load_img`, resizing it to `target_size`.
3. Converts the image to a numpy array and expands its dimensions to fit the model's input shape.

#### Returns
A numpy array of the preprocessed image.

### Function: `predict_celebrity`
#### Purpose
Makes a prediction using the loaded model on an image from an S3 bucket.

#### Parameters
- `model`: The loaded TensorFlow/Keras model.
- `bucket_name`: The name of the S3 bucket.
- `key`: The key (path) of the image in the S3 bucket.
- `threshold`: The threshold for classifying predictions (default is 0.5).

#### Process
1. Preprocesses the image from S3 using `preprocess_image_from_s3`.
2. Makes a prediction using the `model`.
3. Converts the prediction to a binary label (`Celebrity` or `Non-Celebrity`) based on the `argmax` and a threshold.
4. Extracts the confidence of the prediction.

#### Returns
The predicted label (celebrity name) and the confidence of the prediction.

### Prediction Execution
```python
bucket_name = 'celebs3bucket'
input_image_key = 'Sports-celebrity images/Ronaldo/ronaldo_(18).jpg'
predicted_celebrity, confidence = predict_celebrity(loaded_model, bucket_name, input_image_key)
print(f"The predicted label is: {predicted_celebrity} with confidence: {confidence}")
```
- **Prediction**: Calls `predict_celebrity` with the loaded model, S3 bucket name, and image key.
- **Print Results**: Displays the prediction and the confidence level.

## Considerations
- **Model Compatibility**: The model should be compatible with the input image size and format.
- **Label Encoding Logic**: The script assumes a binary classification task and maps the predictions to 'Celebrity' or 'Non-Celebrity'. This logic might need adjustment based on the actual model output and the label encoding used during model training.
- **AWS Credentials**: Proper AWS credentials and permissions are required for accessing S3 buckets.
- **Image Key**: The `input_image_key` should correctly point to the image location within the S3 bucket.
- **Prediction Confidence**: The confidence score provides insight into the model's certainty and should be interpreted carefully, especially in borderline cases.

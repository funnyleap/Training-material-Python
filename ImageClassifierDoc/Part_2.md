# Technical Documentation for Python Script

## Overview
This Python script is designed to interact with an AWS S3 bucket to retrieve images, preprocess these images, and organize them for further processing, likely for machine learning purposes. The script is structured into functions that handle specific tasks: listing files in an S3 bucket, preprocessing images, and loading and preprocessing these images from the bucket.

## Libraries and Modules
- `boto3`: AWS SDK for Python. It provides an interface to Amazon Web Services (AWS), allowing users to access AWS resources directly from Python scripts.
- `numpy` (imported as `np`): Fundamental package for scientific computing in Python. It is used for handling large, multi-dimensional arrays and matrices.
- `PIL` (Python Imaging Library, imported as `Image`): A library for opening, manipulating, and saving many different image file formats.
- `io.BytesIO`: A stream implementation using an in-memory bytes buffer. It can be used as a drop-in replacement for file objects.

## Script Components

### Constants
- `bucket_name`: String. Name of the AWS S3 bucket to interact with.
- `folder_prefix`: String. The prefix within the bucket to specify a folder path.

### Function: `list_files_in_folder`
#### Purpose
Lists all files within a specific folder of an S3 bucket and extracts labels for these files.

#### Parameters
- `bucket_name`: The name of the S3 bucket.
- `folder_prefix`: The folder path within the bucket.

#### Process
1. Initializes an S3 client using `boto3`.
2. Retrieves a list of objects in the specified bucket and folder.
3. Extracts the file path and label for each object.

#### Returns
A list of tuples, each containing the file path and its extracted label.

### Function: `preprocess_image`
#### Purpose
Preprocesses an image for model input.

#### Parameters
- `image`: The image to be processed.
- `target_size`: A tuple for the target size of the image (default is `(224, 224)`).

#### Process
1. Resizes the image to the specified `target_size`.
2. Converts the image to a numpy array and normalizes pixel values to the range [0, 1].

#### Returns
A preprocessed image as a numpy array.

### Function: `load_and_preprocess_images`
#### Purpose
Loads and preprocesses images from an S3 bucket based on provided file paths and labels.

#### Parameters
- `bucket_name`: The name of the S3 bucket.
- `files_labels`: A list of tuples containing file paths and labels.

#### Process
1. Initializes an S3 client using `boto3`.
2. Iterates over the provided `files_labels`.
   - For each file:
     - Retrieves the image data from S3.
     - Opens the image using `PIL.Image`.
     - Preprocesses the image.
     - Appends the processed image and its label to separate lists.
   - Handles exceptions by printing an error message and skipping the file.

#### Returns
Two numpy arrays: one containing the preprocessed image data and the other containing corresponding labels.

### Execution Flow
1. Lists files and their labels in the specified S3 bucket folder.
2. Loads and preprocesses these images.
3. Prints the shapes of the data and labels arrays for verification.

---

# Let's dive into the specific code segments and explain in detail how they function:

### 1. Imports and Global Variables
```python
import boto3
import numpy as np
from PIL import Image
from io import BytesIO

bucket_name = 'celebs3bucket'
folder_prefix = 'Sports-celebrity images/'
```

- **Imports**: The script imports `boto3` for AWS interactions, `numpy` for numerical operations, `PIL.Image` for image processing, and `BytesIO` from the `io` module for handling byte streams.
- **Global Variables**:
  - `bucket_name`: Stores the name of the AWS S3 bucket ('celebs3bucket') to be accessed.
  - `folder_prefix`: Specifies the folder path within the S3 bucket, targeting 'Sports-celebrity images/'.

### 2. Function: `list_files_in_folder`
This function lists all files in a specified S3 bucket folder and extracts labels for each file.
```python
def list_files_in_folder(bucket_name, folder_prefix):
    s3 = boto3.client('s3')
    response = s3.list_objects(Bucket=bucket_name, Prefix=folder_prefix)
    files_labels = [(obj['Key'], obj['Key'].split('/')[1]) for obj in response.get('Contents', []) if '/' in obj['Key']]
    return files_labels
```
- **S3 Client Initialization**: `boto3.client('s3')` initializes an S3 client to interact with the AWS S3 service.
- **Listing Objects**: `s3.list_objects(Bucket=bucket_name, Prefix=folder_prefix)` lists all objects in the specified bucket and folder.
- **Extracting File Paths and Labels**: The function iterates over the contents of the response. For each object, it extracts the 'Key' (the file path) and uses it to create a tuple. The first element of the tuple is the file path, and the second element is the label, extracted by splitting the 'Key' string at '/' and taking the second element. This assumes that the file label is part of the file path.

### 3. Function: `preprocess_image`
This function preprocesses an image by resizing and normalizing it.
```python
def preprocess_image(image, target_size=(224, 224)):
    resized_image = image.resize(target_size)
    array_image = np.array(resized_image) / 255.0  # Normalize pixel values to [0, 1]
    return array_image
```
- **Image Resizing**: `image.resize(target_size)` resizes the input image to the specified target size (default is 224x224 pixels).
- **Normalization**: The image is converted to a numpy array, and its pixel values are normalized to the range [0, 1] by dividing by 255.0 (since pixel values range from 0 to 255).

### 4. Function: `load_and_preprocess_images`
Loads and preprocesses images from the S3 bucket.
```python
def load_and_preprocess_images(bucket_name, files_labels):
    data = []
    labels = []
    s3 = boto3.client('s3')

    for file_key, label in files_labels:
        try:
            response = s3.get_object(Bucket=bucket_name, Key=file_key)
            image_data = response['Body'].read()
            image = Image.open(BytesIO(image_data))
            processed_image = preprocess_image(image)
            data.append(processed_image)
            labels.append(label)
        except Exception as e:
            print(f"Skipping file {file_key} due to error: {e}")

    return np.array(data), np.array(labels)
```
- **Looping Through Files**: Iterates over `files_labels`, which contains tuples of file paths and labels.
- **Image Retrieval and Processing**:
  - Uses `s3.get_object` to fetch each image from the S3 bucket.
  - Reads the image data (`response['Body'].read()`) and loads it into a PIL Image object (`Image.open(BytesIO(image_data))`).
  - The image is then processed using `preprocess_image`.
- **Data Collection**: Processed images and labels are appended to the `data` and `labels` lists, respectively.
- **Exception Handling**: Any errors during processing are caught, and the script prints an error message and continues with the next file.
- **Return Statement**: Converts the `data` and `labels` lists to numpy arrays and returns them.

### 5. Executing the Functions
```python
files_labels = list_files_in_folder(bucket_name, folder_prefix)
data, labels = load_and_preprocess_images(bucket_name, files_labels)
print("Data shape:", data.shape)
print("Labels:", labels)
```
- The `list_files_in_folder` function is called to retrieve file paths and labels.
- `load_and_preprocess_images` is then called to load and preprocess these images.
- Finally, the shapes of the data and labels arrays are printed, providing a quick verification of the data processed.

This script effectively sets up a pipeline for fetching and preprocessing image data from AWS S3, which can be a critical step in various machine learning and data processing workflows, particularly in image-based applications.

---


# [PART - 3 —>](https://github.com/saifeemustafaq/AIYA_December_2023/blob/main/ImageClassifierDoc/Part_3.md)

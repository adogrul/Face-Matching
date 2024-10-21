# VGG16 Image Similarity Finder

This repository provides a simple implementation for finding the most similar image from a folder compared to a user-provided image. It uses the VGG16 pre-trained model to extract feature vectors from images, and then calculates the cosine similarity between the feature vectors to find the closest match.

## Features
- Uses **VGG16** pre-trained model to extract image features.
- Compares the **cosine similarity** of the feature vectors between images.
- Returns and displays the **most similar image** from a given folder.

## Prerequisites

To run the code, you need to have the following libraries installed:

- `tensorflow`
- `opencv-python`
- `numpy`
- `scikit-learn`
- `matplotlib`

You can install these dependencies via `pip`:

```bash
pip install tensorflow opencv-python numpy scikit-learn matplotlib
```
## How It Works
***Load VGG16 Model***: The code uses the VGG16 model pre-trained on the ImageNet dataset. The output layer is modified to retrieve features from the fc1 layer, which is the second fully connected layer.

***Feature Extraction***: Both the user’s image and the images from the folder are resized to 224x224, preprocessed, and passed through the VGG16 model to extract feature vectors.

***Cosine Similarity Calculation:*** The extracted feature vectors are compared using the cosine similarity metric to find the image in the folder that is most similar to the user-provided image.

***Display Results:*** The most similar image, along with the user image, is displayed side by side using matplotlib.

## Code Structure
``load_vgg_model()``: Loads the pre-trained VGG16 model.

``extract_features(model, image)``: Resizes and preprocesses the image, extracts its feature vector using the VGG16 model.

``calculate_cosine_similarity(user_features, hr_features)``: Calculates cosine similarity between the user's image features and a folder image's features.

``load_user_image(filepath)``: Loads the user's image using OpenCV.

``load_similar_images(folder_path)``: Loads all images in a specified folder.

``show_most_similar_image(user_img, best_match_img)``: Displays the user’s image and the most similar image side by side.

``find_most_similar_image(user_image_path, hr_images_folder)``: Main function to find the most similar image from a folder.

## Usage
To use this code, follow the steps below:

### 1. Prepare Your Images:
* Save the image you want to compare as user_image.jpg.
* Place the images you want to search for similarity inside a folder called hr_images (or any folder of your choice).

### 2. Run the Code:

* Example usage to find the most similar image:

```python
user_image_path = 'path/to/user_image.jpg'
hr_images_folder = 'path/to/hr_images_folder'

# Find the most similar image
user_image, best_match_image, best_match_filename, best_similarity = find_most_similar_image(user_image_path, hr_images_folder)

# Display the results
show_most_similar_image(user_image, best_match_image)

print(f"The most similar image is: {best_match_filename} with a similarity score of {best_similarity:.4f}")

```


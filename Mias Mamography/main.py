import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib
from skimage import feature
from sklearn.svm import SVC

def extract_hog_features(image_path, pixels_per_cell=(32, 32), cells_per_block=(2, 2)):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Assuming the image has the same format as in your training data
    # You might need additional preprocessing depending on your specific use case

    # Calculate HOG features for the entire image
    hog_features = feature.hog(image, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)

    return hog_features

# Replace 'path_to_your_image.jpg' with the path to the image you want to test
image_to_test_path = 'C:/Users/CFIX.VN/Documents/DeepLearning/Image Processing/BC-with-MIAs--main/Mias Mamography/imgtest/pic1.jpg'
original_image = cv2.imread(image_to_test_path)
# Extract HOG features for the test image
test_hog_features = extract_hog_features(image_to_test_path)

# Load the trained SVM model
trained_model = joblib.load('C:/Users/CFIX.VN/Documents/DeepLearning/Image Processing/BC-with-MIAs--main/Mias Mamography/svm_model.pkl')  # Replace with the actual path to your trained model

# Reshape the feature vector to match the input shape expected by the SVM model
expected_feature_size = 4356
test_hog_features = test_hog_features.reshape(1, -1)[:,:expected_feature_size]

# Make predictions using the trained model
prediction = trained_model.predict(test_hog_features)

# Display the original image
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# Display the region used for prediction
plt.subplot(1, 3, 2)
x, y, radius = 100, 100, 32  # Adjust these coordinates based on your data
cropped_image = original_image[y - radius:y + radius, x - radius:x + radius]
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.title('Cropped Image for Prediction')

# Display the predicted label
plt.subplot(1, 3, 3)
if prediction[0] == 1:
    prediction_text = 'Malignant'
else:
    prediction_text = 'Benign'
plt.text(0.5, 0.5, f'Predicted: {prediction_text}', horizontalalignment='center', verticalalignment='center', fontsize=12)
plt.axis('off')
plt.title('Prediction Result')

plt.show()

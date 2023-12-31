import os
import cv2
import numpy as np
from skimage import feature
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

# Đường dẫn đến thư mục chứa các hình ảnh JPEG
image_dir = 'D:/Master/HKII/Img Processing/MIAS_Mammography/all-mias/image/'

# Đường dẫn đến tệp 'info.txt'
info_file_path = 'D:/Master/HKII/Img Processing/MIAS_Mammography/all-mias/info.txt'

# Đọc dữ liệu từ tệp 'info.txt'
with open(info_file_path, 'r') as file:
    data = file.readlines()

# Khởi tạo danh sách để lưu trữ thông tin
info_list = []

# Loop qua từng dòng trong dữ liệu
for line in data:
    # Loại bỏ khoảng trắng ở đầu và cuối dòng
    line = line.strip()
    
    # Kiểm tra nếu dòng không rỗng và bắt đầu bằng 'mdb' (để loại bỏ các dòng không liên quan)
    if line and line.startswith('mdb'):
        parts = line.split()
        
        # Kiểm tra nếu có đủ 7 phần từ trong dòng
        if len(parts) >= 7:
            reference_number = parts[0]
            background_tissue = parts[1]
            abnormality = parts[2]
            severity = parts[3]
            
            # Kiểm tra và chuyển đổi các giá trị x, y, radius thành số nguyên
            if parts[4].isdigit() and parts[5].isdigit() and parts[6].isdigit():
                x_coordinate = int(parts[4])
                y_coordinate = int(parts[5])
                radius = int(parts[6])

                info = {
                    'Reference Number': reference_number,
                    'Background Tissue': background_tissue,
                    'Abnormality': abnormality,
                    'Severity': severity,
                    'X Coordinate': x_coordinate,
                    'Y Coordinate': y_coordinate,
                    'Radius': radius
                }
                
                info_list.append(info)

# In ra thông tin đã trích xuất
#for info in info_list:
#    print(info)


# Initialize lists to store HOG features and corresponding labels
hog_features = []
labels = []

# Loop through the list of images and info data
for info in info_list:
    # Build the full path to the JPEG image
    image_path = os.path.join(image_dir, info['Reference Number'] + '.jpg')
    
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale

    # Crop a region around the abnormality using x, y, and radius
    x, y, radius = info['X Coordinate'], info['Y Coordinate'], info['Radius']
    cropped_image = image[y - radius:y + radius, x - radius:x + radius]
    
    # Calculate HOG features for the cropped image
    hog_feature = feature.hog(cropped_image, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
    
    # Append the HOG features to the list
    hog_features.append(hog_feature)
    
    # Append the label (severity) to the labels list
    labels.append(info['Severity'])
    
# Calculate the maximum length of HOG features
max_feature_length = max(len(feature) for feature in hog_features)

# Resize or zero-pad each feature to have the same length
hog_features = [np.pad(feature, (0, max_feature_length - len(feature)), 'constant') if len(feature) < max_feature_length else feature for feature in hog_features]

# Convert the lists to numpy arrays
hog_features = np.array(hog_features)
labels = np.array(labels)

# Split dataset to train and test with test 20%
X_train, X_test, y_train, y_test = train_test_split(hog_features, labels, test_size=0.2, random_state=42)

# Create model SVM with kernel parameters
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Save the final model to a file
model_filename = 'svm_model.pkl'
joblib.dump(best_svm_model, model_filename)
print(f'Model saved to {model_filename}')

# Evaluation model
y_pred = svm_model.predict(X_test)
accuracy_score = accuracy_score(y_test, y_pred)
print(f'Accuracy of the model: {accuracy_score * 100:.2f}%')

import os
import numpy as np
from skimage import io
from skimage import feature
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


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
for info in info_list:
    print(info)
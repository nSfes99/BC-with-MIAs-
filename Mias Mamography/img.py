import os
from PIL import Image

# Đường dẫn đến thư mục chứa tệp .pgm và nơi bạn muốn lưu trữ tệp .jpeg
pgm_directory = 'D:/Master/HKII/Img Processing/MIAS_Mammography/all-mias'
jpeg_directory = 'D:/Master/HKII/Img Processing/MIAS_Mammography/all-mias/image'

# Lặp qua tất cả tệp .pgm trong thư mục đầu vào
for filename in os.listdir(pgm_directory):
    if filename.endswith(".pgm"):
        # Xây dựng đường dẫn đầy đủ đến tệp .pgm
        pgm_path = os.path.join(pgm_directory, filename)
        
        # Đọc hình ảnh .pgm
        image = Image.open(pgm_path)
        # chuyển sang ảnh xám và lưu thành ảnh jpeg
        image = image.convert('L')
        jpeg_filename = os.path.splitext(filename)[0] + ".jpg"
        # Xây dựng đường dẫn
        jpeg_path = os.path.join(jpeg_directory, jpeg_filename)
        image.save(jpeg_path, 'JPEG')

print("Chuyển đổi hoàn tất.")

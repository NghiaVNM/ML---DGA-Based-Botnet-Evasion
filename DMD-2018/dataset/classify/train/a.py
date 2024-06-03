# Mở và đọc file txt cần ghép
with open('./column2.txt', 'r') as file1:
    data = file1.read()

# Mở file txt cần ghép vào và ghi tiếp nội dung
with open('../test/test1/test1label.txt', 'a') as file2:
    file2.write(data)

# Đọc tất cả các dòng từ file
# with open('../test/test1/test1label.txt', 'r') as file:
#     lines = file.readlines()

# # Xóa 1000 dòng đầu tiên
# del lines[0:400000]

# # Ghi lại nội dung vào file
# with open('../test/test1/test1label.txt', 'w') as file:
#     file.writelines(lines)
import pandas as pd

# Đọc dữ liệu từ file CSV
data = pd.read_csv('trainlabel-binary.csv', header=None, sep=';')

# Lấy cột đầu tiên và lưu vào file txt
data.iloc[:, 0].to_csv('binary-train.txt', index=False, header=False)

# Lấy cột thứ hai và lưu vào file txt
data.iloc[:, 1].to_csv('binary-label.txt', index=False, header=False)

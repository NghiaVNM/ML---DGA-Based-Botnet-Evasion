import pandas as pd

def convert_csv_to_lowercase(input_file, output_file):
    # Đọc dữ liệu từ tệp CSV
    data = pd.read_csv(input_file)

    # Chuyển đổi tất cả các ký tự thành chữ thường trong toàn bộ DataFrame
    data = data.apply(lambda x: x.str.lower() if x.dtype == "object" else x)

    # Lưu DataFrame đã được chuyển đổi vào tệp CSV mới
    data.to_csv(output_file, index=False)

# Chuyển đổi và lưu kết quả
convert_csv_to_lowercase("binary-label.txt", "binary-label.txt")

convert_csv_to_lowercase("binary-train.txt", "binary-train.txt")

convert_csv_to_lowercase("test1.txt", "test1.txt")

convert_csv_to_lowercase("test1label.txt", "test1label.txt")

convert_csv_to_lowercase("test2.txt", "test2.txt")

convert_csv_to_lowercase("test2label.txt", "test2label.txt")

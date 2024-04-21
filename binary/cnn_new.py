s1= "a"

# Đọc ký tự từ file đã lưu và lưu vào biến s2
with open("a.txt", "r") as file:
    s2 = file.read().strip()

# So sánh ký tự đã khai báo trong mã Python và ký tự đọc từ file
if s1 == s2:
    print("Hai ký tự giống nhau.")
else:
    print("Hai ký tự khác nhau.")

print("Ký tự trong mã Python:", s1)
print("Ký tự trong file:", s2)

import sqlite3
# Thư viện này có sẵn trong Flask, dùng để băm mật khẩu
from werkzeug.security import generate_password_hash 

# Kết nối tới database (sẽ tạo tệp 'database.db' nếu chưa có)
conn = sqlite3.connect('database.db')
cursor = conn.cursor()

print("Đã kết nối database...")

# --- Bảng 1: Lưu trữ người dùng ---
# KHÔNG BAO GIỜ lưu mật khẩu gốc, luôn luôn băm (hash) mật khẩu
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL
)
''')
print("Đã tạo bảng 'users'.")

# --- Bảng 2: Lưu các quán ăn đã lưu (mối quan hệ 1-nhiều) ---
cursor.execute('''
CREATE TABLE IF NOT EXISTS saved_restaurants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    restaurant_name TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id)
)
''')
print("Đã tạo bảng 'saved_restaurants'.")

# --- (Tùy chọn) Thêm một người dùng mẫu để test ---
try:
    # Băm mật khẩu '123'
    hashed_password = generate_password_hash('123')
    cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                   ('testuser', hashed_password))
    print("Đã thêm user 'testuser' với mật khẩu '123'.")
except sqlite3.IntegrityError:
    print("User 'testuser' đã tồn tại.")


# Lưu thay đổi và đóng kết nối
conn.commit()
conn.close()

print("Hoàn tất khởi tạo database!")
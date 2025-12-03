import os
import json
from flask import Flask, request, jsonify, send_from_directory, g 
from flask_cors import CORS
from utils.system_search_engine import SearchEngine
import pandas as pd

import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash


IMAGE_FOLDER = '/home/truong/Search_Engine_Easy/datas/datas_crawl/food_images'
print(IMAGE_FOLDER)
app = Flask(__name__)
CORS(app)

# --- (MỚI) Thêm cấu hình Database ---
DATABASE = 'database.db' # Tệp này sẽ nằm cùng cấp với app.py

# ==============================================================================
# KHỞI TẠO SEARCH ENGINE
# ==============================================================================
print("🚀 Đang khởi tạo Search Engine... Vui lòng đợi.")
try:
    engine = SearchEngine()
    print("✅ Search Engine đã sẵn sàng nhận yêu cầu.")
except FileNotFoundError as e:
    print(f"💥 LỖI NGHIÊM TRỌNG: {e}")
    engine = None

# ==============================================================================
# (MỚI) CÁC HÀM TRỢ GIÚP KẾT NỐI DATABASE
# ==============================================================================
def get_db():
    """
    Kết nối đến database. Nếu chưa có kết nối, tạo một cái mới.
    """
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        # Cho phép truy cập cột bằng tên (ví dụ: user['password'])
        db.row_factory = sqlite3.Row 
    return db

@app.teardown_appcontext
def close_connection(exception):
    """
    Đóng kết nối database sau khi request kết thúc.
    """
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# ==============================================================================
# ROUTE PHỤC VỤ HÌNH ẢNH (TỪ FILE CỦA BẠN)
# ==============================================================================
@app.route('/images/<path:filename>')
def get_image(filename):
    """
    Phục vụ file ảnh tĩnh từ thư mục 'food_images'.
    """
    print(f"Đang phục vụ ảnh: {filename}")
    return send_from_directory(IMAGE_FOLDER, filename)

# ==============================================================================
# ROUTE TÌM KIẾM (TỪ FILE CỦA BẠN)
# ==============================================================================
@app.route('/search', methods=['GET'])
def search_api():
    print("\n\n=======================================")
    print(f"✅ [app.py] ĐÃ NHẬN ĐƯỢC YÊU CẦU: {request.url}")

    if not engine:
        print("❌ [app.py] LỖI: Engine chưa sẵn sàng.")
        return jsonify({"error": "Search engine chưa được khởi tạo."}), 500

    query = request.args.get('q', '')
    if not query:
        print("❌ [app.py] LỖI: Không có query.")
        return jsonify({"error": "Vui lòng cung cấp query (tham số 'q')."}), 400

    try:
        print(f"🚀 [app.py] BẮT ĐẦU GỌI engine.search(query='{query}') ...")
        results_df = engine.search(query)
        print(f"✅ [app.py] GỌI engine.search() THÀNH CÔNG.")
        
        if results_df.empty:
            print("🟡 [app.py] Kết quả rỗng.")
            return jsonify([])

        results_json = results_df.to_dict('records')
        print(f"✅ [app.py] Đang gửi {len(results_json)} kết quả về trình duyệt.")
        return jsonify(results_json)

    except Exception as e:
        print(f"💥💥💥 [app.py] LỖI NGHIÊM TRỌNG TRONG KHI TÌM KIẾM: {e}")
        return jsonify({"error": "Đã xảy ra lỗi máy chủ nội bộ."}), 500

# ==============================================================================
# (MỚI) CÁC ROUTE CHO VIỆC ĐĂNG KÝ / ĐĂNG NHẬP
# ==============================================================================

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data['username']
    password = data['password']
    
    db = get_db()
    cursor = db.cursor()
    
    # 1. Kiểm tra xem user đã tồn tại chưa
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    existing_user = cursor.fetchone()
    
    if existing_user:
        return jsonify({"success": False, "message": "Tên đăng nhập đã tồn tại."}), 400
        
    # 2. Băm mật khẩu
    hashed_password = generate_password_hash(password)
    
    # 3. Thêm user mới vào DB
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                       (username, hashed_password))
        db.commit()
        return jsonify({"success": True, "message": "Đăng ký thành công!"})
    except Exception as e:
        db.rollback()
        return jsonify({"success": False, "message": f"Lỗi: {e}"}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data['username']
    password = data['password']
    
    db = get_db()
    cursor = db.cursor()
    
    # 1. Tìm user
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    
    if not user:
        return jsonify({"success": False, "message": "Tên đăng nhập không tồn tại."}), 404
        
    # 2. Kiểm tra mật khẩu đã băm
    if check_password_hash(user['password'], password):
        # Đăng nhập thành công!
        return jsonify({
            "success": True, 
            "message": "Đăng nhập thành công!",
            "user_id": user['id'],
            "username": user['username']
        })
    
    else:
        # Sai mật khẩu
        return jsonify({"success": False, "message": "Sai mật khẩu."}), 401

# ==============================================================================
# (MỚI) CÁC ROUTE CHO VIỆC LƯU / BỎ LƯU QUÁN ĂN
# ==============================================================================

@app.route('/get_saved', methods=['GET'])
def get_saved():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"success": False, "message": "Thiếu user_id"}), 400
        
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT restaurant_name FROM saved_restaurants WHERE user_id = ?", (user_id,))
    items = cursor.fetchall()
    
    # Chuyển đổi danh sách các object (Row) thành danh sách các chuỗi (string)
    saved_list = [item['restaurant_name'] for item in items]
    return jsonify({"success": True, "saved_items": saved_list})

@app.route('/save', methods=['POST'])
def save_restaurant():
    data = request.json
    user_id = data['user_id']
    restaurant_name = data['restaurant_name']
    
    db = get_db()
    cursor = db.cursor()
    
    try:
        cursor.execute("INSERT INTO saved_restaurants (user_id, restaurant_name) VALUES (?, ?)", 
                       (user_id, restaurant_name))
        db.commit()
        return jsonify({"success": True, "message": "Đã lưu."})
    except Exception as e:
        db.rollback()
        return jsonify({"success": False, "message": f"Lỗi: {e}"}), 500

@app.route('/unsave', methods=['POST'])
def unsave_restaurant():
    data = request.json
    user_id = data['user_id']
    restaurant_name = data['restaurant_name']
    
    db = get_db()
    cursor = db.cursor()
    
    try:
        cursor.execute("DELETE FROM saved_restaurants WHERE user_id = ? AND restaurant_name = ?", 
                       (user_id, restaurant_name))
        db.commit()
        return jsonify({"success": True, "message": "Đã bỏ lưu."})
    except Exception as e:
        db.rollback()
        return jsonify({"success": False, "message": f"Lỗi: {e}"}), 500

# ==============================================================================
# CHẠY APP (TỪ FILE CỦA BẠN)
# ==============================================================================
if __name__ == '__main__':
    # (Quan trọng!) Nhắc nhở chạy init_db.py
    if not os.path.exists(DATABASE):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"!!! CẢNH BÁO: Không tìm thấy tệp database '{DATABASE}'.")
        print("!!! Bạn CẦN chạy tệp 'init_db.py' MỘT LẦN để tạo database.")
        print("!!! python init_db.py")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
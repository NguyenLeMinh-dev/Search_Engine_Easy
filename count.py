import pandas as pd
import re
import os

# ==============================================================================
# PHẦN 1: CẤU HÌNH
# ==============================================================================

# (INPUT) Đường dẫn đến file QREL cũ (bị lỗi format)
OLD_QREL_FILE = "/home/minh/Documents/SEG_project/datas/all_labels.qrels.csv"

# (OUTPUT) Tên file QREL mới, đã được làm sạch
NEW_QREL_FILE = "/home/minh/Documents/SEG_project/datas/all_labels.qrels.CLEANED.csv"


# ==============================================================================
# PHẦN 2: HÀM CHUẨN HÓA (QUAN TRỌNG NHẤT)
# ==============================================================================

def get_clean_query_id_from_raw_text(raw_query_text):
    """
    Hàm này lấy text thô (ví dụ: "cơm tấm?") và trả về ID đầy đủ.
    """
    
    # 1. Xóa khoảng trắng thừa ở 2 đầu
    sanitized = raw_query_text.strip()
    
    # 2. Thay khoảng trắng bằng gạch dưới
    sanitized = sanitized.replace(' ', '_')
    
    # 3. Loại bỏ TẤT CẢ các ký tự đặc biệt
    sanitized = re.sub(r'[^\w_]', '', sanitized)
    
    # 4. Gộp nhiều gạch dưới liên tiếp
    sanitized = re.sub(r'__+', '_', sanitized)
    
    # 5. (SỬA LỖI) Xóa gạch dưới ở đầu hoặc cuối
    sanitized = sanitized.strip('_')
    
    # 6. Thêm prefix chuẩn
    return f"labeled_search_results_{sanitized}"

# ==============================================================================
# PHẦN 3: LOGIC CHUYỂN ĐỔI
# ==============================================================================

print(f"🚀 Bắt đầu chuẩn hóa file QREL...")
print(f"   Đọc file cũ: {OLD_QREL_FILE}")

try:
    df = pd.read_csv(OLD_QREL_FILE)
except Exception as e:
    print(f"LỖI: Không thể đọc file QREL cũ. Lỗi: {e}")
    exit()

if 'query_id' not in df.columns:
    print("LỖI: File QREL cũ không có cột 'query_id'.")
    exit()

# Lưu lại các query_id cũ để đối chiếu
old_ids = df['query_id'].unique()
print(f"Đã tìm thấy {len(old_ids)} query_id cũ (chưa chuẩn hóa).")

# --- Đây là phần ma thuật ---
# 1. Trích xuất phần "text thô" từ query_id cũ
#    (Ví dụ: "labeled_search_results_Quán_ăn...?" -> "Quán_ăn...?")
df['dirty_query_text'] = df['query_id'].str.replace('labeled_search_results_', '', regex=False)

# 2. Chuyển phần text thô đó về dạng "text gốc" (thay _ thành ' ')
#    (Ví dụ: "Quán_ăn...?" -> "Quán ăn...?")
#    Lưu ý: Đây là một phép "đoán" dựa trên logic cũ, có thể không hoàn hảo
#    nhưng đủ tốt để làm sạch các ký tự đặc biệt.
df['guessed_raw_text'] = df['dirty_query_text'].str.replace('_', ' ')

# 3. Tạo query_id MỚI, SẠCH từ text gốc vừa đoán được
df['clean_query_id'] = df['guessed_raw_text'].apply(get_clean_query_id_from_raw_text)

# 4. Giữ lại các cột quan trọng và đổi tên
final_df = df[['clean_query_id', 'doc_id', 'relevance_score']]
final_df = final_df.rename(columns={'clean_query_id': 'query_id'})

# 5. Lưu file mới
final_df.to_csv(NEW_QREL_FILE, index=False)

print("\n" + "="*50)
print(f"✅ HOÀN TẤT!")
print(f"   File QREL mới đã được lưu tại: {NEW_QREL_FILE}")

# Hiển thị so sánh
new_ids = final_df['query_id'].unique()
print(f"\n   Đã chuẩn hóa thành {len(new_ids)} query_id mới:")
for id in new_ids[:10]: # Chỉ in 10 cái đầu
    print(f"   -> {id}")
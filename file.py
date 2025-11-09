import os

# 1. CẤU HÌNH (Lấy từ file eval_system.py của bạn)
# Đây là thư mục chứa các file "Bài làm" (ví dụ: search_results_com_chay.csv)
RESULT_DIR = "/home/minh/Documents/SEG_project/datas/label" 

# 2. Tiền tố (prefix) của các file kết quả
# (Dựa trên tên file của bạn: "search_results_...")
PREFIX = "labeled_search_results_"

def extract_queries_from_filenames(directory, prefix):
    """
    Quét một thư mục, tìm các tệp có tiền tố,
    và trích xuất query text từ tên tệp.
    """
    query_list = []
    
    print(f"Đang quét thư mục: {directory}\n")
    
    try:
        filenames = [f for f in os.listdir(directory) if f.endswith('.csv')]
        
        if not filenames:
            print(f"Không tìm thấy file .csv nào trong thư mục.")
            return

        for fname in filenames:
            # Chỉ xử lý các tệp "Bài làm" (không phải file "labeled_")
            if fname.startswith(prefix):
                
                # Bỏ tiền tố: "search_results_com_chay.csv" -> "com_chay.csv"
                base_name = fname[len(prefix):]
                
                # Bỏ đuôi .csv: "com_chay.csv" -> "com_chay"
                query_id = os.path.splitext(base_name)[0]
                
                # Thay thế gạch dưới bằng dấu cách: "com_chay" -> "cơm chay"
                # (LƯU Ý: Nếu query gốc có ký tự đặc biệt, nó có thể bị mất)
                # Đây là lý do tại sao dùng query_id (query_01) là tốt nhất
                query_text = query_id.replace('_', ' ')
                
                query_list.append(query_text)
        
        print(f"✅ Đã trích xuất thành công {len(query_list)} truy vấn.")
        
        # In ra ở dạng dễ sao chép (mỗi dòng 1 query)
        print("\n" + "="*40)
        print(" DANH SÁCH TEST QUERIES (dạng text thuần túy)")
        print("="*40)
        for q in query_list:
            print(q)
            
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy thư mục: {directory}")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

# --- Chạy hàm ---
if __name__ == "__main__":
    extract_queries_from_filenames(RESULT_DIR, PREFIX)
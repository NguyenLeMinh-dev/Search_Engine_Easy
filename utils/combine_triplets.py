import pandas as pd
import glob
import os
import sys

# --- CẤU HÌNH ---

# Nơi bạn lưu các file triplet (ví dụ: 'triplets_dataset_0_to_200.csv')
# '.' nghĩa là thư mục hiện tại
Path = os.path.dirname(os.path.abspath(__file__))
TRIPLETS_DIR = os.path.join(Path, '..', 'datas', 'triplets_file')

# Tên file output sau khi gộp
OUTPUT_FILE = os.path.join(Path, '..', 'datas', 'triplets_file','triplets_final_train.csv')

def main():
    # 1. Tìm tất cả các file batch
    search_path = os.path.join(TRIPLETS_DIR, 'triplets_dataset_*.csv')
    all_files = glob.glob(search_path)
    
    if not all_files:
        print(f"LỖI: Không tìm thấy file nào khớp với '{search_path}'.")
        print("Hãy đảm bảo bạn đã chạy 'create_triplets.py' và file nằm đúng thư mục.")
        sys.exit(1)
        
    print(f"Đã tìm thấy {len(all_files)} file triplet. Bắt đầu gộp...")
    
    # 2. Đọc và gộp vào một list
    df_list = []
    for f in all_files:
        try:
            df_list.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            print(f"Cảnh báo: File '{f}' bị rỗng, sẽ bỏ qua.")
        except Exception as e:
            print(f"Lỗi khi đọc file '{f}': {e}")

    if not df_list:
        print("LỖI: Không có dữ liệu để gộp.")
        sys.exit(1)
        
    # 3. Gộp (concatenate)
    full_triplets_df = pd.concat(df_list, ignore_index=True)
    
    # 4. Lưu kết quả
    try:
        full_triplets_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        print(f"\n--- HOÀN TẤT ---")
        print(f"Đã gộp {len(all_files)} file.")
        print(f"Tổng cộng có {len(full_triplets_df)} bộ ba.")
        print(f"Đã lưu kết quả vào file: '{OUTPUT_FILE}'")
    except Exception as e:
        print(f"Lỗi khi lưu file '{OUTPUT_FILE}': {e}")

if __name__ == "__main__":
    main()

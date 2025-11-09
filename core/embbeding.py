import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import sys
import os

# --- 1. CẤU HÌNH ---

# Đường dẫn đến file CSV gốc (chứa text và ID)
DATA_CSV_PATH = '/home/minh/Documents/SEG_project/datas/datas_crawl/final_processed_data.csv'
ID_COLUMN = 'id'
TEXT_COLUMN = 'text_for_embedding'

# Đường dẫn đến thư mục chứa model SBERT ĐÃ FINE-TUNE
FINETUNED_MODEL_PATH = '/home/minh/Documents/SEG_project/datas/triplets_file/fintune_sbert_v1'

# Giới hạn độ dài của model (phải giống lúc huấn luyện)
SBERT_MAX_LENGTH = 256

# Tên file output để lưu embeddings
OUTPUT_EMBEDDING_FILE = 'finetuned_item_embeddings.npy'

# --- 2. HÀM TẢI DỮ LIỆU VÀ MODEL ---

def load_data_and_finetuned_model(data_path, id_col, text_col, model_path, max_length):
    """Tải model fine-tuned và văn bản từ CSV."""

    # Tải model fine-tuned
    print(f"\n--- Đang tải model fine-tuned từ: {model_path} ---")
    try:
        model = SentenceTransformer(model_path)
        model.max_seq_length = max_length
        print(f"Đã ép model.max_seq_length = {max_length}")
    except Exception as e:
        print(f"LỖI: Không thể tải model fine-tuned từ '{model_path}'.")
        print("Hãy đảm bảo bạn đã chạy 'train_finetune.py' thành công và thư mục tồn tại.")
        print(f"Chi tiết lỗi: {e}")
        return None, None # Trả về None nếu không tải được model

    # Tải dữ liệu văn bản
    print(f"\nĐang tải dữ liệu từ: {data_path}")
    try:
        # Đọc ID là string trước để xử lý linh hoạt
        df = pd.read_csv(data_path, dtype={id_col: str})
        df = df[[id_col, text_col]].dropna()
        df = df.drop_duplicates(subset=[id_col])
        # Ép kiểu ID sang INT (nếu bạn muốn ID là int trong map)
        # df[id_col] = df[id_col].astype(int) # Bỏ comment nếu muốn ID là int
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file data '{data_path}'")
        return model, None # Trả về model đã tải nhưng không có text
    except Exception as e:
        print(f"Lỗi khi đọc file data hoặc ép kiểu ID: {e}")
        return model, None # Trả về model đã tải nhưng không có text

    texts = df[text_col].astype(str).tolist()
    # ids = df[id_col].tolist() # Lấy ID nếu cần map với embedding sau này

    print(f"Đã tải {len(texts)} văn bản.")
    # Chỉ cần trả về model và texts cho việc tạo embedding
    return model, texts

# --- 3. HÀM CHÍNH (MAIN) ---

def main():
    # 1. Tải model fine-tuned và dữ liệu văn bản
    model, texts = load_data_and_finetuned_model(
        DATA_CSV_PATH,
        ID_COLUMN,
        TEXT_COLUMN,
        FINETUNED_MODEL_PATH,
        SBERT_MAX_LENGTH
    )

    # Kiểm tra xem model và text có tải thành công không
    if model is None or texts is None:
        sys.exit(1) # Dừng nếu có lỗi

    # 2. Mã hóa văn bản bằng model fine-tuned
    print(f"\nBắt đầu mã hóa {len(texts)} văn bản bằng model fine-tuned...")
    try:
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    except Exception as e:
        print(f"Lỗi khi mã hóa văn bản: {e}")
        sys.exit(1)

    print(f"Mã hóa hoàn tất. Shape của embeddings: {embeddings.shape}")

    # 3. Lưu kết quả embedding
    try:
        np.save(OUTPUT_EMBEDDING_FILE, embeddings)
        print(f"\n--- HOÀN TẤT ---")
        print(f"Đã lưu {embeddings.shape[0]} vector embedding vào file:")
        print(f"  -> {OUTPUT_EMBEDDING_FILE}")
    except Exception as e:
        print(f"Lỗi khi lưu file embedding '{OUTPUT_EMBEDDING_FILE}': {e}")
        sys.exit(1)

# --- 4. CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    main()
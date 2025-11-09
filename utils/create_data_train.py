import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os

""" ----------------CONFIG-------------------"""
PATH = os.path.dirname(os.path.abspath(__file__))
DATA_CSV_PATH = os.path.join(PATH, '..', 'datas', 'datas_crawl', 'final_processed_data.csv')
FINAL_TRIPLET_FILE = os.path.join(PATH, '..', 'datas', 'triplets_file', 'triplets_final_train.csv')
SBERT_MODEL_NAME = 'dangvantuan/vietnamese-embedding'
SBERT_MAX_LENGTH = 256# Giới hạn của model
OUTPUT_MODEL_PATH = os.path.join(PATH, '..', 'datas', 'triplets_file', 'fintune_sbert_v1')
""" -----------------------------------------"""

""" ----------------PARAMS-------------------"""
BATCH_SIZE = 16  # Giảm nếu bạn bị lỗi CUDA Out of Memory (OOM)
NUM_EPOCHS = 1   # Huấn luyện 1 epoch trước, nếu tốt thì tăng lên
MARGIN = 0.5     # Margin cho TripletLoss (0.5 là khởi đầu tốt)

# --- 2. HÀM CHUẨN BỊ DỮ LIỆU ---
def prepare_data(triplets_df, data_csv_path):
    """
    Tải văn bản và chuyển đổi triplets thành list các InputExample.
    """
    print(f"Đang tải văn bản từ '{data_csv_path}'...")
    try:
        # SỬA 1: Đọc file data và ép kiểu ID thành INT
        data_df = pd.read_csv(data_csv_path)
        # Xử lý nếu ID là '000001' (string) hoặc 1 (int)
        data_df['id'] = data_df['id'].astype(str).astype(int) 
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file CSV dữ liệu chính: '{data_csv_path}'")
        return None
    except Exception as e:
        print(f"Lỗi khi đọc file data.csv: {e}")
        return None

    # Tạo một "Từ điển" (map) để tra cứu text từ ID
    print("Đang tạo bản đồ (map) ID -> Text...")
    text_map = pd.Series(
        data_df['text_for_embedding'].values, 
        index=data_df['id'] # ID bây giờ là INT (ví dụ: 1, 2, 3)
    ).to_dict()
    
    print("Đang tạo các 'InputExample' cho huấn luyện...")
    train_examples = []
    
    # SỬA 2: Ép kiểu ID của triplet thành INT
    triplets_df['anchor_id'] = triplets_df['anchor_id'].astype(str).astype(int)
    triplets_df['positive_id'] = triplets_df['positive_id'].astype(str).astype(int)
    triplets_df['negative_id'] = triplets_df['negative_id'].astype(str).astype(int)
    
    for row in tqdm(triplets_df.itertuples(), total=len(triplets_df), desc="Chuẩn bị dữ liệu"):
        # Lấy văn bản từ map (Bây giờ get(1) sẽ khớp với key 1)
        anchor_text = text_map.get(row.anchor_id)
        positive_text = text_map.get(row.positive_id)
        negative_text = text_map.get(row.negative_id)
        
        # Kiểm tra nếu ID nào đó không tìm thấy text (do lỗi)
        if not all([anchor_text, positive_text, negative_text]):
            # print(f"Cảnh báo: Bỏ qua triplet vì thiếu text cho ID: {row.anchor_id}, {row.positive_id}, {row.negative_id}")
            continue
            
        # InputExample là lớp chuẩn của SBERT cho triplet loss
        train_examples.append(
            InputExample(texts=[anchor_text, positive_text, negative_text])
        )
        
    print(f"Đã tạo {len(train_examples)} mẫu huấn luyện.")
    return train_examples

# --- 3. HÀM HUẤN LUYỆN ---
def main():
    # 1. Tải dữ liệu triplet đã gộp
    print(f"Đang tải dữ liệu triplet đã gộp từ: '{FINAL_TRIPLET_FILE}'")
    try:
        triplets_df = pd.read_csv(FINAL_TRIPLET_FILE)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file '{FINAL_TRIPLET_FILE}'.")
        print("Bạn hãy chạy file 'merge_triplets.py' trước.")
        sys.exit(1)
    except Exception as e:
        print(f"Lỗi khi đọc file triplet đã gộp: {e}")
        sys.exit(1)

    # 2. Chuẩn bị dữ liệu (tạo InputExamples)
    train_examples = prepare_data(triplets_df, DATA_CSV_PATH)
    if not train_examples:
        print("LỖI: Không có dữ liệu huấn luyện nào được tạo.")
        sys.exit(1)

    # 3. Tải mô hình SBERT gốc
    print(f"Đang tải mô hình SBERT gốc: '{SBERT_MODEL_NAME}'...")
    model = SentenceTransformer(SBERT_MODEL_NAME)
    
    # Sửa lỗi CUDA: Ép model phải cắt văn bản
    model.max_seq_length = SBERT_MAX_LENGTH
    print(f"Đã ép model.max_seq_length = {SBERT_MAX_LENGTH}")

    # 4. Định nghĩa DataLoader
    print("Đang tạo DataLoader...")
    train_dataloader = DataLoader(
        train_examples, 
        shuffle=True, 
        batch_size=BATCH_SIZE,
        num_workers=8,
        pin_memory=True
    )

    # 5. Định nghĩa Hàm Loss
    # Dùng TripletLoss với khoảng cách Cosine (tốt cho text)
    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=MARGIN
    )
    
    # 6. Huấn luyện mô hình
    print(f"\n--- BẮT ĐẦU HUẤN LUYỆN (FINE-TUNE) ---")
    print(f"Model: {SBERT_MODEL_NAME}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Margin: {MARGIN}")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=NUM_EPOCHS,
        warmup_steps=100, # Số bước khởi động
        output_path=OUTPUT_MODEL_PATH,
        show_progress_bar=True,
        use_amp=True,
        optimizer_params={'lr': 2e-5},
        scheduler='WarmupLinear'
    )
    
    print(f"\n--- HUẤN LUYỆN HOÀN TẤT ---")
    print(f"Mô hình đã được fine-tune và lưu tại: '{OUTPUT_MODEL_PATH}'")

if __name__ == "__main__":
    main()

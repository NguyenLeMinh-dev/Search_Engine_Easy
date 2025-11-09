import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os
import argparse

# ==============================================================================
# SECTION 1: CẤU HÌNH TRUNG TÂM (ĐỂ ĐÁNH GIÁ)
# ==============================================================================

# *** THAY ĐỔI MODEL TẠI ĐÂY ***
# Chỉ cần thay đổi dòng này, các file output sẽ tự động cập nhật tên.
MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
# Ví dụ thay đổi:
# MODEL_NAME = "vinai/phobert-large"
# MODEL_NAME = "nguyenvulebinh/envibert" 

# --- Cấu hình đường dẫn ---
# (Giữ nguyên đường dẫn tuyệt đối từ file của bạn)
BASE_DATA_PATH = "/home/minh/Documents/SEG_project/datas/datas_crawl/"
INPUT_CSV = os.path.join(BASE_DATA_PATH, "final_processed_data.csv")

# --- Tự động tạo tên file output dựa trên MODEL_NAME ---
# Lấy phần cuối của tên model làm slug (ví dụ: "phobert-base-v2")
model_slug = MODEL_NAME.split('/')[-1]
# File .pt để lưu trữ tensor đã tokenize
OUTPUT_TENSORS_PATH = os.path.join(BASE_DATA_PATH, f"{model_slug}.pt")
# File .npy để lưu trữ embeddings cuối cùng
OUTPUT_EMBEDDINGS_PATH = os.path.join(BASE_DATA_PATH, f"{model_slug}.npy")

# --- Cấu hình Tokenizer ---
MAX_LENGTH = 256
TOKENIZE_BATCH_SIZE = 32   # Batch size cho việc tokenize

# --- Cấu hình Embedding ---
EMBED_BATCH_SIZE = 128 # Batch size để tạo embedding (tăng cho GPU mạnh)

# ==============================================================================
# SECTION 2: HÀM TOKENIZE (TỪ Tokenize.py)
# ==============================================================================

def tokenize_for_phobert():
    """
    Chạy pipeline tokenization.
    Đọc từ file CSV và lưu kết quả ra file .pt.
    """
    # --- 1. Load the clean dataset ---
    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"📖 Đã đọc {len(df)} dòng từ '{INPUT_CSV}'.")
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file '{INPUT_CSV}'. Vui lòng chạy script 'clean_data.py' trước.")
        return False # Trả về False nếu thất bại

    if 'text_for_embedding' not in df.columns:
        print(f"❌ Lỗi: Không tìm thấy cột 'text_for_embedding' trong file CSV.")
        return False

    # --- 2. Load the PhoBERT tokenizer ---
    print(f"🤖 Đang tải Tokenizer cho model ('{MODEL_NAME}')...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        print("✅ Tải tokenizer thành công.")
    except Exception as e:
        print(f"❌ Lỗi khi tải tokenizer: {e}. Vui lòng kiểm tra kết nối mạng.")
        return False

    # --- 3. Tokenize the text in batches ---
    texts = df['text_for_embedding'].tolist()
    print(f"\n⚙️  Bắt đầu tokenize {len(texts)} dòng văn bản (Batch size = {TOKENIZE_BATCH_SIZE})...")
    
    all_input_ids = []
    all_attention_masks = []

    for i in tqdm(range(0, len(texts), TOKENIZE_BATCH_SIZE), desc="Tokenizing batches"):
        batch_texts = texts[i:i + TOKENIZE_BATCH_SIZE]
        
        tokenized_batch = tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
        
        all_input_ids.append(tokenized_batch['input_ids'])
        all_attention_masks.append(tokenized_batch['attention_mask'])

    input_ids = torch.cat(all_input_ids, dim=0)
    attention_mask = torch.cat(all_attention_masks, dim=0)

    tokenized_output = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    print("✅ Tokenize hoàn tất!")

    # --- 4. Save the results ---
    input_ids_shape = tokenized_output['input_ids'].shape
    print(f"\nKích thước của tensor 'input_ids': {input_ids_shape}")

    try:
        torch.save(tokenized_output, OUTPUT_TENSORS_PATH)
        print(f"\n💾 Đã lưu kết quả vào file: '{OUTPUT_TENSORS_PATH}'")
        print(f"👉 Bước tiếp theo: Dùng file này để chạy 'generate_embeddings'.")
        return True # Trả về True nếu thành công
    except Exception as e:
        print(f"❌ Lỗi khi lưu file: {e}")
        return False

# ==============================================================================
# SECTION 3: HÀM TẠO EMBEDDING (TỪ generate_embedding.py)
# ==============================================================================

def generate_embeddings():
    """
    Chạy pipeline tạo embedding.
    Đọc từ file .pt và lưu kết quả ra file .npy.
    """
    # --- 1. Setup device (GPU/CPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Sử dụng thiết bị: {device}")
    if device.type == 'cpu':
        print("⚠️  Cảnh báo: Chạy trên CPU sẽ chậm hơn đáng kể.")

    # --- 2. Load tokenized data ---
    try:
        # Sử dụng đường dẫn OUTPUT_TENSORS_PATH đã được cấu hình
        tokenized_data = torch.load(OUTPUT_TENSORS_PATH)
        dataset = TensorDataset(tokenized_data['input_ids'], tokenized_data['attention_mask'])
        dataloader = DataLoader(dataset, batch_size=EMBED_BATCH_SIZE, shuffle=False)
        print(f"💾 Đã tải dữ liệu đã tokenize từ '{OUTPUT_TENSORS_PATH}'.")
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file '{OUTPUT_TENSORS_PATH}'.")
        print("👉 Vui lòng chạy bước 'tokenize' trước: python your_script_name.py --step tokenize")
        return False

    # --- 3. Load the Model ---
    print(f"🤖 Đang tải mô hình ('{MODEL_NAME}')...")
    try:
        model = AutoModel.from_pretrained(MODEL_NAME).to(device)
        model.eval()
        print("✅ Tải mô hình thành công.")
    except Exception as e:
        print(f"❌ Lỗi khi tải mô hình: {e}. Vui lòng kiểm tra kết nối mạng.")
        return False
        
    # --- 4. Generate Embeddings in Batches ---
    all_embeddings = []
    print(f"\n⚙️  Bắt đầu tạo embeddings với batch size = {EMBED_BATCH_SIZE}...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Embeddings"):
            b_input_ids, b_attention_mask = [b.to(device) for b in batch]
            
            outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask)
            
            # Lấy [CLS] token embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            
            all_embeddings.append(cls_embeddings.cpu().numpy())

    final_embeddings = np.concatenate(all_embeddings, axis=0)
    print("✅ Tạo embeddings hoàn tất!")
    print(f"Kích thước của ma trận embeddings: {final_embeddings.shape}")

    # --- 5. Save the final embeddings ---
    try:
        # Sử dụng đường dẫn OUTPUT_EMBEDDINGS_PATH đã được cấu hình
        np.save(OUTPUT_EMBEDDINGS_PATH, final_embeddings)
        print(f"\n💾 Đã lưu embeddings vào file: '{OUTPUT_EMBEDDINGS_PATH}'")
        print("👉 Bước tiếp theo: Dùng file này để chạy 'search_engine.py'.")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi lưu file: {e}")
        return False

# ==============================================================================
# SECTION 4: SCRIPT EXECUTION (Trình điều khiển Pipeline)
# ==============================================================================

if __name__ == "__main__":
    # Thêm trình phân tích đối số để chọn bước chạy
    parser = argparse.ArgumentParser(
        description=f"Pipeline Tokenize và Tạo Embedding cho model {MODEL_NAME}."
    )
    parser.add_argument(
        '--step', 
        type=str, 
        default='all', 
        choices=['all', 'tokenize', 'embed'],
        help="Bước cần chạy: 'tokenize' (chỉ tokenize), 'embed' (chỉ tạo embedding), hoặc 'all' (cả hai, mặc định)."
    )
    args = parser.parse_args()

    print(f"🚀 BẮT ĐẦU PIPELINE CHO MODEL: {MODEL_NAME}")
    print(f"   Input CSV: {INPUT_CSV}")
    print(f"   Output Tensors: {OUTPUT_TENSORS_PATH}")
    print(f"   Output Embeddings: {OUTPUT_EMBEDDINGS_PATH}")
    
    if args.step in ['all', 'tokenize']:
        print("\n" + "="*50)
        print("BƯỚC 1: TOKENIZE DỮ LIỆU")
        print("="*50)
        success_tokenize = tokenize_for_phobert()
        if not success_tokenize:
            print("❌ Dừng pipeline do lỗi ở bước Tokenize.")
            exit() # Thoát nếu bước 1 lỗi
            
    if args.step in ['all', 'embed']:
        print("\n" + "="*50)
        print("BƯỚC 2: TẠO EMBEDDINGS")
        print("="*50)
        success_embed = generate_embeddings()
        if not success_embed:
            print("❌ Dừng pipeline do lỗi ở bước Embedding.")
            exit() # Thoát nếu bước 2 lỗi

    print("\n🎉 Pipeline hoàn tất thành công! 🎉")
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
import json
import joblib
import argparse
import os
import sys
from tqdm import tqdm

# Tắt cảnh báo SettingWithCopyWarning (tùy chọn)
pd.options.mode.chained_assignment = None

# --- Hằng số ---
# !!! SỬA LẠI ĐÂY: Trỏ đến model SBERT ĐÃ FINE-TUNE
FINETUNED_SBERT_MODEL = '/home/minh/Documents/SEG_project/datas/triplets_file/fintune_sbert_v1'
# Giới hạn độ dài của model (phải giống lúc huấn luyện)
SBERT_MAX_LENGTH = 256

# --- 1. HÀM XỬ LÝ VĂN BẢN (S-BERT) - Đã cập nhật ---
def process_sbert_finetuned(texts, model_path, max_length):
    """
    Mã hóa văn bản bằng model S-BERT ĐÃ FINE-TUNE.
    """
    print(f"\nĐang tải mô hình S-BERT FINE-TUNED từ: {model_path}...")
    try:
        model = SentenceTransformer(model_path)
        # Ép độ dài tối đa để đảm bảo nhất quán
        model.max_seq_length = max_length
        print(f"Đã ép model.max_seq_length = {max_length}")
    except Exception as e:
        print(f"Lỗi khi tải model S-BERT fine-tuned: {e}")
        print("Vui lòng đảm bảo thư mục model tồn tại và đúng định dạng.")
        sys.exit(1)

    print("Bắt đầu mã hóa văn bản (S-BERT Fine-tuned)...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32  # Có thể tăng nếu GPU đủ mạnh
    )
    print(f"Mã hóa S-BERT hoàn tất. Kích thước: {embeddings.shape}")
    return embeddings

# --- 2. HÀM XỬ LÝ ĐẶC TRƯNG SỐ (Giữ nguyên) ---
def process_numerical(df):
    """
    Thực hiện kỹ thuật đặc trưng và chuẩn hóa cho các cột số.
    """
    print("\nĐang xử lý đặc trưng số (Numerical)...")
    # Tạo bản sao để tránh SettingWithCopyWarning
    df_copy = df.copy()

    # Kỹ thuật đặc trưng
    df_copy['price_avg'] = (df_copy['price_min'] + df_copy['price_max']) / 2
    df_copy['price_range'] = df_copy['price_max'] - df_copy['price_min']
    df_copy['duration'] = (df_copy['close_hour'] - df_copy['open_hour'] + 24) % 24

    numerical_cols = [
        'rating', 'price_avg', 'price_range',
        'open_hour', 'close_hour', 'duration',
        'gps_lat', 'gps_long'
    ]

    # Kiểm tra sự tồn tại của cột trước khi xử lý
    existing_cols = [col for col in numerical_cols if col in df_copy.columns]
    missing_cols = [col for col in numerical_cols if col not in df_copy.columns]
    if missing_cols:
        print(f"Cảnh báo: Thiếu các cột số sau, sẽ bỏ qua: {', '.join(missing_cols)}")

    if not existing_cols:
        print("Lỗi: Không có cột số nào hợp lệ để xử lý.")
        # Trả về mảng rỗng và scaler rỗng nếu không có cột nào
        return np.array([]).reshape(len(df_copy), 0), None

    # Xử lý NaN
    for col in existing_cols:
        if df_copy[col].isnull().any():
            print(f"Cảnh báo: Cột '{col}' có giá trị null. Sẽ điền bằng giá trị trung bình.")
            mean_val = df_copy[col].mean()
            df_copy[col] = df_copy[col].fillna(mean_val)

    # Chuẩn hóa
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_copy[existing_cols])

    print(f"Xử lý số hoàn tất. Kích thước: {scaled_data.shape}")
    return scaled_data, scaler


# --- 3. HÀM XỬ LÝ ĐẶC TRƯNG DANH MỤC (Giữ nguyên) ---
def process_categorical(df):
    """
    Xây dựng từ điển và ánh xạ các cột danh mục sang chỉ số (integer index).
    """
    print("\nĐang xử lý đặc trưng danh mục (Categorical)...")
    # Tạo bản sao để tránh SettingWithCopyWarning
    df_copy = df.copy()
    vocabs = {}

    # City & District
    for col in ['city', 'district']:
        if col not in df_copy.columns:
            print(f"Cảnh báo: Thiếu cột danh mục '{col}'. Bỏ qua.")
            continue
        df_copy[col] = df_copy[col].fillna('UNKNOWN')
        unique_values = df_copy[col].unique()
        vocab = {name: i + 1 for i, name in enumerate(unique_values)}
        vocab['UNKNOWN'] = 0
        df_copy[f'{col}_index'] = df_copy[col].map(vocab).fillna(0).astype(int)
        vocabs[col] = vocab
        print(f"Đã xây dựng từ điển cho '{col}' với {len(vocab)} mục.")

    # Tags
    if 'tags' not in df_copy.columns:
        print("Cảnh báo: Thiếu cột 'tags'. Bỏ qua xử lý tags.")
        df_copy['tag_indices'] = [[] for _ in range(len(df_copy))] # Tạo cột rỗng
    else:
        all_tags = set()
        for tag_list in df_copy['tags'].dropna():
            tags = [tag.strip() for tag in tag_list.split(',')]
            all_tags.update(tags)

        tag_vocab = {tag: i + 1 for i, tag in enumerate(sorted(list(all_tags)))} # Sắp xếp để đảm bảo thứ tự
        tag_vocab['UNKNOWN_TAG'] = 0
        vocabs['tag'] = tag_vocab
        print(f"Đã xây dựng từ điển cho 'tags' với {len(tag_vocab)} mục.")

        def tags_to_indices(tag_string, vocab):
            if pd.isna(tag_string): return []
            tags = [tag.strip() for tag in tag_string.split(',')]
            return [vocab.get(tag, vocab['UNKNOWN_TAG']) for tag in tags]

        df_copy['tag_indices'] = df_copy['tags'].apply(lambda x: tags_to_indices(x, tag_vocab))

    # Chỉ giữ lại các cột ID và cột chỉ số cần thiết
    categorical_cols_to_keep = ['id']
    if 'city_index' in df_copy.columns: categorical_cols_to_keep.append('city_index')
    if 'district_index' in df_copy.columns: categorical_cols_to_keep.append('district_index')
    if 'tag_indices' in df_copy.columns: categorical_cols_to_keep.append('tag_indices')

    categorical_df = df_copy[categorical_cols_to_keep]

    return categorical_df, vocabs

# --- HÀM LƯU KẾT QUẢ (Giữ nguyên) ---
def save_artifacts(output_dir, item_ids, sbert_embeddings, numerical_data, numerical_scaler, categorical_df, vocabs):
    """
    Lưu tất cả các đối tượng đã xử lý vào thư mục đầu ra.
    Thêm item_ids để đảm bảo thứ tự.
    """
    print(f"\nĐang lưu kết quả vào thư mục: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # 0. Lưu danh sách ID (để đảm bảo thứ tự)
    # Chuyển ID sang kiểu string trước khi lưu
    item_ids_str = [str(i) for i in item_ids]
    pd.DataFrame({'id': item_ids_str}).to_csv(os.path.join(output_dir, 'item_ids.csv'), index=False)

    # 1. Lưu S-BERT
    if sbert_embeddings.size > 0:
        np.save(os.path.join(output_dir, 'item_sbert_embeddings.npy'), sbert_embeddings)
    else: print("Cảnh báo: Không có SBERT embedding để lưu.")

    # 2. Lưu Dữ liệu Số
    if numerical_data.size > 0:
        np.save(os.path.join(output_dir, 'item_numerical_features.npy'), numerical_data)
    else: print("Cảnh báo: Không có dữ liệu số để lưu.")

    # 3. Lưu Scaler
    if numerical_scaler:
        joblib.dump(numerical_scaler, os.path.join(output_dir, 'numerical_scaler.pkl'))
    else: print("Cảnh báo: Không có scaler số để lưu.")

    # 4. Lưu Dữ liệu Danh mục
    if not categorical_df.empty:
        # Chuyển 'tag_indices' (list) thành chuỗi JSON để lưu CSV
        if 'tag_indices' in categorical_df.columns:
            categorical_df['tag_indices'] = categorical_df['tag_indices'].apply(json.dumps)
        # Chuyển ID sang string trước khi lưu
        categorical_df['id'] = categorical_df['id'].astype(str)
        categorical_df.to_csv(os.path.join(output_dir, 'item_categorical_indices.csv'), index=False, encoding='utf-8')
    else: print("Cảnh báo: Không có dữ liệu danh mục để lưu.")

    # 5. Lưu Từ điển
    for name, vocab in vocabs.items():
        vocab_path = os.path.join(output_dir, f'{name}_vocab.json')
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

    print("\n--- HOÀN TẤT XỬ LÝ ITEM DATA ---")
    print("Các tệp đã được tạo trong thư mục:", output_dir)

# --- HÀM CHÍNH (MAIN) ---
def main(args):
    """
    Hàm điều phối chính để chạy toàn bộ quy trình.
    """
    # 1. Tải dữ liệu
    print(f"Đang tải dữ liệu gốc từ: {args.input_csv}")
    try:
        df = pd.read_csv(args.input_csv)
        # Ép kiểu ID sang INT để xử lý nhất quán, xử lý lỗi nếu có
        try:
             df['id'] = df['id'].astype(str).astype(int)
        except Exception as e:
             print(f"Cảnh báo: Không thể chuyển cột 'id' sang int. Sử dụng kiểu string. Lỗi: {e}")
             df['id'] = df['id'].astype(str) # Giữ lại là string nếu lỗi

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp '{args.input_csv}'")
        sys.exit(1)
    except Exception as e:
        print(f"Lỗi khi đọc CSV: {e}")
        sys.exit(1)

    print(f"Tải thành công {len(df)} hàng.")

    # Đảm bảo cột 'text_for_embedding' là chuỗi và xử lý NaN
    df['text_for_embedding'] = df['text_for_embedding'].fillna('').astype(str)

    # Lấy danh sách ID gốc để đảm bảo thứ tự
    original_item_ids = df['id'].tolist()

    # 2. Chạy các pipeline
    sbert_embeddings = process_sbert_finetuned(
        df['text_for_embedding'].tolist(),
        FINETUNED_SBERT_MODEL,
        SBERT_MAX_LENGTH
    )
    numerical_data, numerical_scaler = process_numerical(df)
    categorical_df, vocabs = process_categorical(df)

    # Đảm bảo tất cả các mảng/df có cùng số lượng hàng với ID gốc
    if not (len(sbert_embeddings) == len(numerical_data) == len(categorical_df) == len(original_item_ids)):
         print("\nLỗi nghiêm trọng: Số lượng hàng không khớp sau khi xử lý!")
         print(f"  - SBERT embeddings: {len(sbert_embeddings)}")
         print(f"  - Numerical data: {len(numerical_data)}")
         print(f"  - Categorical data: {len(categorical_df)}")
         print(f"  - Original IDs: {len(original_item_ids)}")
         sys.exit(1)

    # Sắp xếp lại categorical_df theo original_item_ids trước khi lưu
    # Điều này cực kỳ quan trọng nếu các hàm xử lý làm thay đổi thứ tự
    categorical_df = categorical_df.set_index('id').loc[original_item_ids].reset_index()


    # 3. Lưu kết quả
    save_artifacts(
        args.output_dir,
        original_item_ids, # Truyền danh sách ID gốc
        sbert_embeddings,
        numerical_data,
        numerical_scaler,
        categorical_df,
        vocabs
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu Item Tower (V2 - Dùng Fine-tuned SBERT).")
    parser.add_argument(
        '--input_csv',
        type=str,
        default='/home/minh/Documents/SEG_project/datas/datas_crawl/final_processed_data.csv', # Đặt giá trị mặc định
        help="Đường dẫn đến tệp data.csv đầu vào."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/minh/Documents/SEG_project/datas/datas_crawl/processed_item_data', # Đặt tên thư mục output mặc định
        help="Thư mục để lưu các tệp Item Tower đã xử lý."
    )

    args = parser.parse_args()
    main(args)

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import argparse
from tqdm import tqdm # Thêm tqdm để theo dõi tiến trình
import json
import os
import sys

# --- 1. CẤU HÌNH ---
NUM_USERS = 2000        # Số lượng user giả lập (như bạn yêu cầu)
MIN_INTERACTIONS = 10   # Số lượng tương tác tối thiểu cho mỗi user
MAX_INTERACTIONS = 70   # Số lượng tương tác tối đa cho mỗi user
START_DATE = "2024-01-01"
END_DATE = "2025-10-31"

PATH_DATA = os.path.abspath(os.path.join(__file__, "../../../datas"))
ITEM_DATA_CSV = os.path.join(PATH_DATA, 'datas_crawl/final_processed_data.csv')


PATH_DATA = os.path.abspath(os.path.join(__file__, "../../data"))
VOCAB_PATH = os.path.join(PATH_DATA, 'processed_item_data/tag_vocab.json')

OUTPUT_FILE = os.path.join(PATH_DATA, 'persona_user_interactions.csv') 

# Cấu hình "Persona"
INTENTIONAL_RATIO = 0.8 # 80% tương tác là "có chủ đích" (theo gu)
MAX_FAVORITE_TAGS = 3   # Mỗi user sẽ có tối đa 3 tag yêu thích

# --- 2. HÀM HỖ TRỢ ---

def load_data_and_maps(item_csv_path, vocab_path):
    """Tải dữ liệu item, tag vocab, và tạo các map cần thiết."""
    print(f"Đang tải dữ liệu item từ: {item_csv_path}")
    try:
        df_items = pd.read_csv(item_csv_path)
        # Ép kiểu ID (ví dụ: 1)
        df_items['id'] = df_items['id'].astype(str).astype(int)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy tệp '{item_csv_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"Lỗi khi đọc file Item CSV: {e}")
        sys.exit(1)

    print(f"Đang tải tag vocab từ: {vocab_path}")
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            tag_vocab = json.load(f)
            # Lấy danh sách tag (bỏ 'UNKNOWN_TAG' ở index 0)
            valid_tags = [tag for tag, idx in tag_vocab.items() if idx != 0]
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy tệp '{vocab_path}'.")
        print("Vui lòng chạy 'create_item_data_v2.py' trước để tạo vocab.")
        sys.exit(1)

    print("Đang xây dựng map Item -> Tags và Tag -> Items...")
    item_id_to_tags = {} # Map: item_id -> set(tags)
    tag_to_item_ids = {tag: [] for tag in valid_tags} # Map: tag -> list(item_ids)

    # Lọc bỏ các hàng không có tag
    df_items_tagged = df_items.dropna(subset=['tags'])

    for _, row in df_items_tagged.iterrows():
        item_id = row['id']
        tags = set(tag.strip() for tag in row['tags'].split(','))
        item_id_to_tags[item_id] = tags
        
        for tag in tags:
            if tag in tag_to_item_ids:
                tag_to_item_ids[tag].append(item_id)

    all_item_ids = df_items['id'].unique().tolist()
    print("Xây dựng map hoàn tất.")
    
    return all_item_ids, valid_tags, tag_to_item_ids

def create_persona(valid_tags, max_tags):
    """Tạo "persona" cho user bằng cách chọn ngẫu nhiên các tag yêu thích."""
    num_tags = random.randint(1, max_tags)
    favorite_tags = set(random.sample(valid_tags, num_tags))
    return favorite_tags

def get_preferred_item_pool(favorite_tags, tag_to_item_ids):
    """Lấy danh sách các item_id khớp với các tag yêu thích."""
    pool = set()
    for tag in favorite_tags:
        pool.update(tag_to_item_ids.get(tag, []))
    return list(pool)

def generate_timestamp(start_date_str, end_date_str):
    """Tạo một timestamp ngẫu nhiên trong khoảng thời gian."""
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    time_delta = end_date - start_date
    random_seconds = random.uniform(0, time_delta.total_seconds())
    return start_date + timedelta(seconds=random_seconds)

# --- 3. HÀM TẠO DỮ LIỆU CHÍNH ---

def generate_persona_data(all_item_ids, valid_tags, tag_to_item_ids):
    """Tạo dữ liệu tương tác người dùng dựa trên "persona"."""
    
    all_interactions = []

    print(f"Bắt đầu tạo dữ liệu cho {NUM_USERS} users...")
    for i in tqdm(range(NUM_USERS), desc="Tạo User Interactions"):
        user_id = f"user_{i+1:04d}" # user_0001
        
        # 1. Tạo Persona
        favorite_tags = create_persona(valid_tags, MAX_FAVORITE_TAGS)
        preferred_item_pool = get_preferred_item_pool(favorite_tags, tag_to_item_ids)
        
        # Nếu user này thích tag hiếm và không có item nào,
        # họ sẽ hoạt động như user "khám phá" (chỉ click ngẫu nhiên)
        if not preferred_item_pool:
            preferred_item_pool = all_item_ids # Dùng tất cả item

        # 2. Tạo Lịch sử Tương tác
        num_interactions = random.randint(MIN_INTERACTIONS, MAX_INTERACTIONS)
        
        for _ in range(num_interactions):
            interaction = {"user_id": user_id}
            
            # Chọn loại tương tác (giả lập tỷ lệ thực tế)
            interaction_type = random.choices(
                ["view_detail", "rating", "bookmark"], 
                weights=[0.75, 0.15, 0.10], 
                k=1
            )[0]
            interaction["interaction_type"] = interaction_type
            
            # 3. Chọn Item (Quy tắc 80/20)
            is_intentional = random.random() < INTENTIONAL_RATIO
            
            if is_intentional:
                # 80% - Chọn từ "gu"
                item_id = random.choice(preferred_item_pool)
                if interaction_type == "rating":
                    # Cho điểm cao vì đúng gu
                    interaction["rating_value"] = random.choice([4, 5]) 
            else:
                # 20% - Khám phá ngẫu nhiên
                item_id = random.choice(all_item_ids)
                if interaction_type == "rating":
                    # Cho điểm ngẫu nhiên, có thể thấp
                    interaction["rating_value"] = random.choice([1, 2, 3, 4]) 
            
            interaction["item_id"] = item_id
            interaction["timestamp"] = generate_timestamp(START_DATE, END_DATE)
            
            all_interactions.append(interaction)

    print(f"\nĐã tạo tổng cộng {len(all_interactions)} tương tác.")
    return pd.DataFrame(all_interactions)

# --- 4. HÀM CHÍNH (MAIN) ---
def main(args):
    """Hàm điều phối chính."""
    # 1. Tải dữ liệu item và các map
    all_item_ids, valid_tags, tag_to_item_ids = load_data_and_maps(
        args.item_data_csv,
        args.vocab_path
    )
    
    # 2. Tạo dữ liệu giả lập
    fake_df = generate_persona_data(all_item_ids, valid_tags, tag_to_item_ids)

    # 3. Sắp xếp theo thời gian (giống thực tế)
    fake_df['timestamp'] = pd.to_datetime(fake_df['timestamp'])
    fake_df = fake_df.sort_values(by="timestamp").reset_index(drop=True)
    
    # Ép kiểu lại ID item (đã là int) về string để nhất quán
    fake_df['item_id'] = fake_df['item_id'].astype(str)

    # 4. Lưu kết quả
    print(f"Đang lưu dữ liệu tương tác vào: {OUTPUT_FILE}")
    try:
        # Điền NaN cho cột 'rating_value' (cho các event 'view_detail')
        fake_df['rating_value'] = fake_df['rating_value'].fillna(0).astype(int)
        
        # Sắp xếp lại cột
        output_cols = ['user_id', 'item_id', 'interaction_type', 'rating_value', 'timestamp']
        # Lấy các cột tồn tại trong df
        final_cols = [col for col in output_cols if col in fake_df.columns]
        
        fake_df[final_cols].to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        print(f"--- HOÀN TẤT ---")
        print(f"Đã lưu dữ liệu vào '{OUTPUT_FILE}' với {len(fake_df)} tương tác.")
        print("File này chứa các cột: user_id, item_id, interaction_type, rating_value, timestamp.")

    except Exception as e:
        print(f"Lỗi khi lưu file CSV: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo dữ liệu tương tác người dùng (có Persona).")
    parser.add_argument(
        '--item_data_csv',
        type=str,
        default=ITEM_DATA_CSV,
        help="Đường dẫn đến tệp data.csv chứa ID và tags."
    )
    parser.add_argument(
        '--vocab_path',
        type=str,
        default=VOCAB_PATH,
        help="Đường dẫn đến file tag_vocab.json."
    )
    args = parser.parse_args()
    
    # Đảm bảo các đường dẫn đầu vào tồn tại
    if not os.path.exists(args.item_data_csv):
        print(f"Lỗi: Không tìm thấy file data item: {args.item_data_csv}")
        sys.exit(1)
    if not os.path.exists(args.vocab_path):
        print(f"Lỗi: Không tìm thấy file vocab tag: {args.vocab_path}")
        print("Vui lòng chạy 'create_item_data_v2.py' trước.")
        sys.exit(1)
        
    main(args)

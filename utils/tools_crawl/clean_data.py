import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import re
import os
from PIL import Image
from io import BytesIO
import unidecode
import concurrent.futures

# ==============================================================================
# SECTION 1: CONSTANTS AND CONFIGURATION
# ==============================================================================

# --- Input/Output Files ---
INPUT_CSV = "/home/minh/Documents/SEG_project/datas/foody_cantho_with_tags.csv"
OUTPUT_CSV = "final_processed_data.csv"
IMAGE_FOLDER = "food_images"
COMMENT_CHAR_LIMIT = 400 # Giảm nhẹ giới hạn để chừa chỗ cho ngữ nghĩa mới

# ==============================================================================
# SECTION 2: HELPER FUNCTIONS FOR DATA CLEANING
# ==============================================================================

# --- Các hàm clean cơ bản (giữ nguyên, đã rất tốt) ---

def clean_text(text):
    if pd.isna(text): return ''
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\sÀ-ỹ,.-]', '', text)
    return text

def clean_comment_text(text):
    if pd.isna(text): return ''
    text = str(text).strip().replace('\n', '. ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\sÀ-ỹ.,!?]', '', text)
    return text

def remove_accents(text):
    """Chuyển văn bản về không dấu, lowercase. Hàm này sẽ được dùng nhiều."""
    return unidecode.unidecode(str(text)).lower().strip()

def clean_price(price_str):
    if pd.isna(price_str): return None, None
    s = str(price_str).replace('đ', '').replace('.', '').replace(',', '').strip()
    numbers = re.findall(r'\d+', s)
    if len(numbers) == 1:
        val = float(numbers[0])
        return val, val
    elif len(numbers) >= 2:
        return float(numbers[0]), float(numbers[1])
    return None, None

def clean_rating(r):
    try:
        val = float(r)
        return round(val, 2) if val <= 10 else round(val / 10, 2)
    except (ValueError, TypeError): return None

def clean_open_close(t):
    times = re.findall(r'\d{1,2}:\d{2}', str(t))
    if len(times) >= 2:
        try:
            open_h = int(times[0].split(':')[0])
            close_h = int(times[1].split(':')[0])
            # Xử lý trường hợp qua đêm (ví dụ: 18:00 - 02:00)
            if close_h < open_h:
                close_h += 24
            return float(open_h), float(close_h)
        except (ValueError, IndexError): return None, None
    return None, None

def clean_gps(gps_str):
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(gps_str))
    if len(nums) >= 2: return float(nums[0]), float(nums[1])
    return None, None

def download_image_worker(args):
    url, folder, size, img_id = args
    if not isinstance(url, str) or not url.strip(): return None
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert('RGB')
            img = img.resize(size)
            filename = f"{img_id:06d}.jpg"
            path = os.path.join(folder, filename)
            img.save(path, "JPEG", quality=90)
            return path.replace("\\", "/")
    except Exception: return None
    return None

# ==============================================================================
# SECTION 3: NÂNG CẤP - HÀM TẠO NGỮ NGHĨA
# ==============================================================================

def generate_price_semantics(price_min, price_max):
    """Tạo văn bản ngữ nghĩa từ thông tin giá cả."""
    if pd.isna(price_min):
        return ''
    if price_max <= 50000:
        return 'Giá rẻ. Phù hợp học sinh sinh viên.'
    if price_min <= 100000 and price_max <= 250000:
        return 'Giá cả hợp lý, bình dân.'
    if price_min > 250000:
        return 'Phân khúc cao cấp, sang trọng.'
    return 'Giá cả đa dạng.'

def generate_time_semantics(open_h, close_h):
    """Tạo văn bản ngữ nghĩa từ thời gian mở cửa."""
    if pd.isna(open_h):
        return ''
    phrases = []
    if open_h <= 8:
        phrases.append('phục vụ bữa sáng')
    if open_h <= 12 and close_h >= 13:
        phrases.append('bán buổi trưa')
    if open_h <= 18 and close_h >= 19:
        phrases.append('bán buổi tối')
    if close_h >= 22:
        phrases.append('có bán khuya')
    if (close_h - open_h) >= 12:
        phrases.append('mở cửa cả ngày')
    
    return 'Thời gian: ' + ', '.join(phrases) + '.' if phrases else ''

def generate_rating_semantics(rating):
    """Tạo văn bản ngữ nghĩa từ điểm đánh giá."""
    if pd.isna(rating):
        return ''
    if rating >= 9.0:
        return 'Chất lượng xuất sắc, đánh giá rất cao.'
    if rating >= 8.0:
        return 'Quán ngon, chất lượng tốt.'
    if rating >= 7.0:
        return 'Địa điểm khá, được yêu thích.'
    return ''

# ==============================================================================
# SECTION 4: NÂNG CẤP - HÀM TẠO VĂN BẢN CHO SEARCH
# ==============================================================================

def create_embedding_text(row):
    """
    NÂNG CẤP: Tạo văn bản CÓ DẤU cho PhoBERT (Semantic Search).
    Bao gồm các cụm từ ngữ nghĩa đã được tạo ra.
    """
    parts = []
    
    # 1. Tên quán
    parts.append(f"{row['name']}.")
    
    # 2. Tags (tín hiệu ngữ nghĩa rõ ràng nhất)
    if pd.notna(row['tags']) and row['tags']:
        parts.append(f"Thể loại: {row['tags']}.")
    
    # 3. Ngữ nghĩa được tạo ra (MỚI)
    sem_price = generate_price_semantics(row['price_min'], row['price_max'])
    if sem_price: parts.append(sem_price)
    
    sem_time = generate_time_semantics(row['open_hour'], row['close_hour'])
    if sem_time: parts.append(sem_time)
        
    sem_rating = generate_rating_semantics(row['rating'])
    if sem_rating: parts.append(sem_rating)

    # 4. Bình luận (đã được rút gọn)
    if pd.notna(row['comments']) and row['comments']:
        truncated_comments = row['comments'][:COMMENT_CHAR_LIMIT]
        if len(row['comments']) > COMMENT_CHAR_LIMIT:
            truncated_comments += "..."
        parts.append(f"Một số đánh giá: {truncated_comments}")

    # 5. Thông tin phụ: Địa chỉ
    parts.append(f"Địa chỉ tại {row['address']}.")
    
    return ' '.join(parts)

def create_bm25_text(row):
    """
    MỚI: Tạo văn bản KHÔNG DẤU cho BM25 (Keyword Search).
    Đây là cột "Phiên bản Dữ liệu Song song" chúng ta đã thảo luận.
    """
    parts = [
        row['name'],
        row['tags'],
        row['comments'], # Dùng bình luận gốc để có nhiều từ khóa
        row['address'],
        # Thêm các ngữ nghĩa đã tạo (không dấu) để tăng cường từ khóa
        generate_price_semantics(row['price_min'], row['price_max']),
        generate_time_semantics(row['open_hour'], row['close_hour']),
        generate_rating_semantics(row['rating'])
    ]
    
    full_text = ' '.join([str(p) for p in parts if pd.notna(p) and p])
    return remove_accents(full_text) # Sử dụng hàm remove_accents

# ==============================================================================
# SECTION 5: MAIN PROCESSING FUNCTION (ĐÃ CẬP NHẬT)
# ==============================================================================

def main():
    """
    Main function to run the entire data cleaning and processing pipeline.
    """
    # --- 1. Read the source CSV file ---
    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"📖 Đã đọc {len(df)} dòng từ '{INPUT_CSV}'.")
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file '{INPUT_CSV}'. Vui lòng chạy script 'patch_tags_scraper.py' trước.")
        return

    # --- 2. Clean and Standardize Data ---
    print("✨ Bắt đầu làm sạch và chuẩn hóa dữ liệu...")
    df['name'] = df['name'].apply(clean_text)
    df['address'] = df['address'].apply(clean_text)
    df['rating'] = df['rating'].apply(clean_rating)
    df['comments'] = df['comments'].apply(clean_comment_text)
    df['tags'] = df['tags'].apply(clean_text)

    df[['price_min', 'price_max']] = df['price'].apply(lambda x: pd.Series(clean_price(x)))
    df[['open_hour', 'close_hour']] = df['open_close'].apply(lambda x: pd.Series(clean_open_close(x)))
    df[['gps_lat', 'gps_long']] = df['gps'].apply(lambda x: pd.Series(clean_gps(x)))

    # --- 3. Download Images in Parallel ---
    df.dropna(subset=['name', 'image_src'], inplace=True)
    df = df.reset_index(drop=True)
    
    print(f"🖼️  Đang tải {len(df)} ảnh (chạy song song)...")
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    tasks = [(row['image_src'], IMAGE_FOLDER, (224, 224), i + 1) for i, row in df.iterrows()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(download_image_worker, tasks), total=len(tasks), desc="Downloading images"))
    df['image_path'] = results
    
    df.dropna(subset=['image_path'], inplace=True)
    df = df.reset_index(drop=True)
    print(f"✅ Đã tải thành công {len(df)} ảnh.")

    # --- 4. CẬP NHẬT: Create Enriched Text Columns ---
    print("✍️  Tạo các cột văn bản giàu ngữ cảnh (Nâng cấp)...")
    
    # 4a. Tạo cột 'text_for_embedding' (CÓ DẤU, giàu ngữ nghĩa)
    df['text_for_embedding'] = df.apply(create_embedding_text, axis=1)
    
    # 4b. Tạo cột 'text_for_bm25' (KHÔNG DẤU, đầy đủ từ khóa)
    df['text_for_bm25'] = df.apply(create_bm25_text, axis=1)

    # 4c. Tạo cột 'name_no_accent' (dùng cho gợi ý hoặc tìm kiếm nhanh)
    df['name_no_accent'] = df['name'].apply(remove_accents)
    
    # --- 5. Trích xuất thông tin phụ ---
    def extract_district(address):
        match = re.search(r'Quận\s+([\w\s]+)', str(address), re.IGNORECASE)
        return match.group(1).strip() if match else 'Khác'
    df['district'] = df['address'].apply(extract_district)
    df['city'] = 'Cần Thơ'

    # --- 6. Finalize and Save ---
    df['id'] = [f"{i:06d}" for i in range(1, len(df) + 1)]
    final_cols = [
        'id', 'name', 'name_no_accent', 'tags', 'address', 'district', 'city', 'rating',
        'price_min', 'price_max', 'open_hour', 'close_hour', 'gps_lat', 'gps_long',
        'text_for_embedding', # Cột cho PhoBERT (có dấu)
        'text_for_bm25',      # Cột cho BM25 (không dấu)
        'image_path', 'comments', 'url'
    ]
    
    existing_cols = [col for col in final_cols if col in df.columns]
    df_final = df[existing_cols]

    df_final.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n🎉 HOÀN TẤT! {len(df_final)} dòng dữ liệu sạch đã được lưu tại '{OUTPUT_CSV}'")

# ==============================================================================
# SECTION 6: SCRIPT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    main()
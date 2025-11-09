import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import re

# ==============================================================================
# SECTION 1: CONSTANTS AND CONFIGURATION
# ==============================================================================

# --- Input/Output Files ---
INPUT_CSV = "foody_cantho.csv"
OUTPUT_CSV = "foody_cantho_with_tags.csv"
NUM_TO_TEST = None # Set to None to run all rows: `NUM_TO_TEST = None`

# --- Culinary Dictionary ---
# This dictionary is the "brain" that helps identify meaningful tags.

# General categories from the website's filters
FOOD_CATEGORIES = [
    "Sang trọng", "Buffet", "Nhà hàng", "Ăn vặt/vỉa hè", "Ăn chay", 
    "Café/Dessert", "Quán ăn", "Bar/Pub", "Quán nhậu", "Beer club", 
    "Tiệm bánh", "Tiệc tận nơi", "Shop Online", "Giao cơm văn phòng", "Khu Ẩm Thực"
]

# Cuisine types from the website's filters
CUISINE_TYPES = [
    "Món Bắc", "Món Trung Hoa", "Món Miền Trung", "Món Miền Nam", "Món Ấn Độ",
    "Món Thái", "Ý", "Pháp", "Đức", "Món Nhật", "Món Hàn", "Thụy sĩ", "Singapore", 
    "Mỹ", "Đài Loan", "Bánh Pizza", "Đặc sản vùng"
]

# An expanded list of common Vietnamese dishes and keywords to improve recognition
COMMON_DISHES_KEYWORDS = [
    # Main dishes & Noodles
    "bún", "phở", "cơm", "mì", "lẩu", "bánh mì", "xôi", "cháo", "hủ tiếu", "miến", 
    "bún riêu", "bún mắm", "bún thịt nướng", "mì quảng", "cao lầu", "bún bò huế", 
    "bún bò", "bún chả", "bún đậu", "cơm tấm", "cơm rang", "cơm gà", "mì cay",
    # Various types of cakes
    "bánh xèo", "bánh khọt", "bánh canh", "bánh tráng", "bánh cuốn", "bánh bèo", 
    "bánh bột lọc", "bánh ngọt", "bánh bao", "bánh flan", "bánh giò", "bánh pía",
    # Snacks & Appetizers
    "ăn vặt", "gỏi cuốn", "nem nướng", "nem chua rán", "phá lấu", "trứng vịt lộn", 
    "bột chiên", "gỏi", "súp", "salad", "khoai tây chiên", "xúc xích", "xiên que",
    "há cảo", "sủi cảo", "chả giò", "chạo tôm", "gà giòn", "gà rán", "mì ý", "spaghetti",
    
    # Main ingredients
    "gà", "vịt", "bò", "heo", "cá", "tôm", "cua", "ghẹ", "hải sản", "ốc", "ếch", 
    "trứng", "nem", "chả", "xá xíu", "sườn", "giò", "pate", "phô mai",
    # Desserts & Drinks
    "chè", "trà sữa", "cà phê", "cafe", "sinh tố", "nước ép", "kem", "sữa chua", 
    "tàu hủ", "sâm bổ lượng", "rau má", "nước mía", "trà đào", "trà chanh", "matcha",
    # Styles & Cooking methods
    "chay", "dinh dưỡng", "quay", "chiên", "xào", "nướng", "hấp", "luộc", "thập cẩm", 
    "đặc biệt", "bình dân", "nhậu", "mắm tôm", "sushi", "pizza", "bbq", "steak"
]

# ==============================================================================
# SECTION 2: HELPER FUNCTIONS
# ==============================================================================

def build_tag_dictionary(lists_of_tags):
    """
    Creates a comprehensive set of valid, standardized tags from multiple lists.
    This set will be used as our dictionary for matching.
    """
    valid_tags = set()
    for tag_list in lists_of_tags:
        for item in tag_list:
            # Split items like "Café/Dessert" into "Café" and "Dessert"
            parts = re.split(r'[/()]', item)
            for part in parts:
                part = part.strip().lower()
                if part:
                    valid_tags.add(part)
    return valid_tags

def extract_meaningful_tags(text, dictionary):
    """
    Scans a block of text and extracts phrases (of 1, 2, or 3 words)
    that exist in our culinary dictionary.
    """
    found_tags = set()
    # Split text more robustly (by space, comma, hyphen, etc.)
    words = [word for word in re.split(r'[\s,-/&]+', text.lower()) if word]
    
    # Scan for 3-word, then 2-word, then 1-word phrases to prioritize longer, more specific tags
    for n in (3, 2, 1):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i+n])
            if phrase in dictionary:
                found_tags.add(phrase)
    return found_tags

def scrape_all_tags(session, url, dictionary):
    """
    Performs the 3-step strategy to comprehensively gather tags for a given URL.
    1. Harvest: Get existing tags from the page.
    2. Infer: Extract keywords from the restaurant's name.
    3. Discover: Find keywords from the menu.
    """
    try:
        response = session.get(url, timeout=10)
        if response.status_code != 200: return ""
        soup = BeautifulSoup(response.text, 'html.parser')
        
        full_text_to_analyze = ""

        # Step 1: Harvest existing tags
        category_container = soup.select_one(".main-info-title .category")
        if category_container:
            full_text_to_analyze += " " + category_container.get_text(" ", strip=True)

        # Step 2: Infer from the restaurant name
        name_element = soup.select_one("h1[itemprop='name']")
        if name_element:
            full_text_to_analyze += " " + name_element.get_text(strip=True)

        # Step 3: Discover from the menu (up to 10 items)
        menu_items = soup.select(".delivery-dishes-group .delivery-dishes-item .title-name")[:10]
        for item in menu_items:
            full_text_to_analyze += " " + item.get_text(strip=True)
            
        # Final step: Match all collected text against our dictionary
        final_tags = extract_meaningful_tags(full_text_to_analyze, dictionary)
        
        # Capitalize and sort for clean output
        capitalized_tags = [tag.capitalize() for tag in final_tags]
        return ", ".join(sorted(list(set(capitalized_tags))))
        
    except Exception:
        return ""

# ==============================================================================
# SECTION 3: MAIN EXECUTION
# ==============================================================================

def main():
    """
    Main function to run the entire tag patching process.
    """
    # Build the master dictionary of tags
    valid_tags_dictionary = build_tag_dictionary([FOOD_CATEGORIES, CUISINE_TYPES, COMMON_DISHES_KEYWORDS])
    print(f"📖 Đã xây dựng từ điển hoàn chỉnh với {len(valid_tags_dictionary)} tag hợp lệ.")

    # --- Read the source CSV file ---
    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"📂 Đã đọc {len(df)} dòng từ '{INPUT_CSV}'.")
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file '{INPUT_CSV}'. Vui lòng chạy script cào dữ liệu chính trước.")
        return

    # --- Prepare for scraping ---
    df_to_process = df.head(NUM_TO_TEST).copy() if NUM_TO_TEST is not None else df.copy()
    if NUM_TO_TEST is not None:
        print(f"🔬 Chế độ TEST: Chỉ xử lý {len(df_to_process)} quán ăn đầu tiên.")

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })

    if 'tags' not in df.columns:
        df['tags'] = ''

    # --- Start the scraping loop ---
    print(f"\n🏷️  Bắt đầu quá trình 'vá' tags cho {len(df_to_process)} quán ăn...")
    for index, row in tqdm(df_to_process.iterrows(), total=df_to_process.shape[0], desc="Updating tags"):
        url = row['url']
        tags = scrape_all_tags(session, url, valid_tags_dictionary)
        
        # Update the tag directly in the original DataFrame
        df.loc[index, 'tags'] = tags
        time.sleep(0.1) # Small delay to be polite to the server

    # --- Save the enriched data ---
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n🎉 HOÀN TẤT! Đã lưu dữ liệu được làm giàu vào file: '{OUTPUT_CSV}'")
    if NUM_TO_TEST is not None:
        print(f"Lưu ý: Chỉ {NUM_TO_TEST} dòng đầu tiên được cập nhật tags mới.")

if __name__ == "__main__":
    main()


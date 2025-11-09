from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.edge.options import Options
import time
import re
import csv



def click_load_more_until_end(driver, timeout=5, max_tries=1):
    """
    Tự động click nút 'Xem thêm' trên trang danh sách Foody đến khi hết.
    - timeout: thời gian tối đa chờ mỗi lần (giây)
    - max_tries: giới hạn số lần thử để tránh vòng lặp vô hạn
    """
    click_count = 0
    while click_count < max_tries:
        try:
            # Tìm nút 'Xem thêm'
            button = WebDriverWait(driver, timeout).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "a.fd-btn-more"))
            )

            # Cuộn đến nút
            driver.execute_script("arguments[0].scrollIntoView({behavior:'smooth', block:'center'});", button)
            time.sleep(0.5)

            # Click bằng JavaScript cho chắc
            driver.execute_script("arguments[0].click();", button)
            click_count += 1
            print(f"👉 Click 'Xem thêm' lần {click_count}")

            # Đợi nội dung tải xong
            time.sleep(3)

        except Exception:
            print("✅ Đã load hết tất cả quán ăn — không còn nút 'Xem thêm'.")
            break



# =============================
# HÀM LẤY VÀ CHỌN LỌC BÌNH LUẬN
# =============================
def scrape_and_select_comments(driver):
    """
    Cào tất cả bình luận, phân loại chúng theo điểm số, và chọn ra một bộ
    đại diện nhất (tối đa 4-5 bình luận) để làm giàu ngữ cảnh.
    """
    all_comments = []


    try:
        # BƯỚC 1: TÌM CONTAINER LỚN (ĐÃ XÁC NHẬN TỪ ẢNH)
        # SỬA LỖI: Dùng By.CLASS_NAME thay vì By.ID
        review_list_container = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CLASS_NAME, "review-list")) 
        )
        
        # BƯỚC 2: TÌM CÁC KHỐI BÌNH LUẬN CON (ĐÃ XÁC NHẬN TỪ ẢNH)
        # Class "review-item" đã chính xác.
        comment_blocks = review_list_container.find_elements(By.CLASS_NAME, "review-item")

        print(f"    -> Đã tìm thấy {len(comment_blocks)} khối bình luận trên trang.")

        for block in comment_blocks:
            try:
                comment_text = block.find_element(By.CSS_SELECTOR, "div.rd-des span.ng-binding").text.strip()
                rating_text = block.find_element(By.CSS_SELECTOR, "div.review-points span.ng-binding").text.strip()
                rating = float(rating_text)
                
                if comment_text:
                    all_comments.append({"rating": rating, "text": comment_text})
            except Exception:
                continue
                
    except Exception:
        print("    -> Không tìm thấy container chứa bình luận ('review-list').")
        return ""

    # --- LOGIC LỰA CHỌN THÔNG MINH (Giữ nguyên) ---
    if not all_comments:
        return ""

    positive = sorted([c for c in all_comments if c['rating'] >= 7.0], key=lambda x: x['rating'], reverse=True)
    neutral = [c for c in all_comments if 5.0 <= c['rating'] < 7.0]
    negative = sorted([c for c in all_comments if c['rating'] < 5.0], key=lambda x: x['rating'])

    selected_comments = []
    selected_comments.extend(positive[:3])
    if negative:
        selected_comments.extend(negative[:1])
        
    if len(selected_comments) < 4 and neutral:
        selected_comments.extend(neutral[:(4 - len(selected_comments))])
    
    final_text = ". ".join([comment['text'] for comment in selected_comments])
    
    return final_text

# =============================
# 1️⃣  KHỞI TẠO WEBDRIVER
# =============================

edge_options = Options()
edge_options.add_argument("user-data-dir=/home/minh/Documents/selenium_profile")
# edge_options.add_argument("--start-maximized")
edge_options.add_argument("profile-directory=Default")
service = Service(r"/home/minh/Documents/Selenium/msedgedriver")
 
driver = webdriver.Edge(service=service, options =edge_options)


# =============================
# 2️⃣  LẤY DANH SÁCH LINK QUÁN ĂN (Giữ nguyên)
# =============================
start_url = "https://www.foody.vn/can-tho"
driver.get(start_url)
time.sleep(3)

click_load_more_until_end(driver)

print("Đang cuộn trang để tải thêm địa điểm...")
for _ in range(3):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

elements = driver.find_elements(By.TAG_NAME, "a")
links = set()
for el in elements:
    href = el.get_attribute("href")
    if not href:
        continue
    href = href.strip()
    
    # CẢI TIẾN LOGIC LỌC LINK
    if href.startswith(f"{start_url}/") and "/binh-luan" not in href and len(href.split('/')) == 5:
         # THÊM ĐIỀU KIỆN: Loại bỏ chính xác link trang chủ có dạng ".../can-tho/"
         if href != f"{start_url}/" and not any(x in href for x in ["/food/", "/su-kien", "/bo-suu-tap", "/bai-viet", "/video", "/khuyen-mai", "/coupon", "/o-dau", "/top-thanh-vien","/hinh-anh"]):
            links.add(href)

links = list(links)
print(f"\nTìm thấy tổng cộng: {len(links)} link quán ăn.")

# =============================
# 3️⃣  TRUY CẬP TỪNG LINK VÀ LẤY DỮ LIỆU
# =============================
data = []
for i, url in enumerate(links, 1):  # Tăng lên 10 để test được nhiều trường hợp hơn
    print(f"\n[{i}/{len(links)}] Đang crawl: {url}")
    driver.get(url)
    time.sleep(3)

    # CẢI TIẾN: Bọc từng mục trong try-except riêng để không bỏ lỡ dữ liệu
    
    # Tên quán (bắt buộc phải có)
    try:
        name = driver.find_element(By.CSS_SELECTOR, "h1[itemprop='name']").text
    except Exception as e:
        print(f"    -> Lỗi: Không tìm thấy tên quán. Bỏ qua link này. Lỗi: {e}")
        continue # Nếu không có tên, bỏ qua luôn

    # Địa chỉ
    try:
        address_street = driver.find_element(By.CSS_SELECTOR, 'span[itemprop="streetAddress"]').text
        address_locality = driver.find_element(By.CSS_SELECTOR, 'span[itemprop="addressLocality"]').text
        address_region = driver.find_element(By.CSS_SELECTOR, 'span[itemprop="addressRegion"]').text
        address_full = f"{address_street}, {address_locality}, {address_region}"
    except:
        address_full = "" # Nếu lỗi thì gán giá trị rỗng

    # Điểm đánh giá
    try:
        rating = driver.find_element(By.CSS_SELECTOR, "div[itemprop='ratingValue']").text
    except:
        rating = ""

    # Giờ mở cửa
    try:
        hours_text = driver.find_element(By.CSS_SELECTOR, '.micro-timesopen span:nth-of-type(3)').get_attribute('innerText').strip()
    except:
        hours_text = ""

    # Giá
    try:
        price = driver.find_element(By.CSS_SELECTOR, 'span[itemprop="priceRange"]').get_attribute('innerText').strip()
    except:
        price = ""

    # GPS
    try:
        lat = driver.find_element(By.CSS_SELECTOR, 'meta[property="place:location:latitude"]').get_attribute("content")
        lon = driver.find_element(By.CSS_SELECTOR, 'meta[property="place:location:longitude"]').get_attribute("content")
        gps = f"{lat}, {lon}"
    except:
        gps = ""
        
    # Link ảnh
    try:
        image_src = driver.find_element(By.CSS_SELECTOR, "div.main-image img").get_attribute("src")
    except:
        image_src = ""

    # Bình luận (hàm này đã tự xử lý lỗi)
    comments = scrape_and_select_comments(driver)
    
    # Thêm dữ liệu vào danh sách
    data.append({
        "name": name,
        "address": address_full,
        "rating": rating,
        "open_close": hours_text,
        "price": price,
        "gps": gps,
        "image_src": image_src,
        "comments": comments,
        "url": url
    })
    print(f"    -> Lấy dữ liệu thành công cho: {name}")

# =============================
# 4️⃣  LƯU KẾT QUẢ RA FILE CSV (Giữ nguyên)
# =============================
if data:
    output_file = "foody_cantho.csv"
    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    print(f"\n✅ Đã lưu {len(data)} dòng dữ liệu vào file: {output_file}")
else:
    print("\n⚠️ Không có dữ liệu nào được cào, không tạo file CSV.")

# =============================
driver.quit()


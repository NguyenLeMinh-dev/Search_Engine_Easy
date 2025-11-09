import pandas as pd
import google.generativeai as genai
import os
import json
import time
from tqdm import tqdm

# ==============================================================================
# PHẦN 1: CẤU HÌNH
# ==============================================================================

API_KEY = "AIzaSyBDquurKfKANJDXlnA8-pvCbyOfBCDzAXs"

if not API_KEY:
    raise EnvironmentError("Lỗi: Vui lòng đặt biến môi trường GEMINI_API_KEY")

# Sử dụng model bạn đang chạy, ví dụ: gemini-1.5-pro-latest
MODEL_NAME = "gemini-2.5-flash-lite" 
genai.configure(api_key=API_KEY)

INPUT_DIR = "/home/minh/Documents/SEG_project/datas/queries_lable"
OUTPUT_DIR = "/home/minh/Documents/SEG_project/datas/label"
RATE_LIMIT_DELAY = 3.1 

generation_config = {
    "temperature": 0.0,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 100,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    generation_config=generation_config,
    safety_settings=safety_settings
)

# ==============================================================================
# PHẦN 2: HỆ THỐNG PROMPT VÀ DÁN NHÃN
# ==============================================================================

def create_prompt(query, document_text):
    """Tạo prompt chi tiết cho LLM."""
    
    # === SỬA LỖI 1: Thêm cặp ngoặc nhọn {{ và }} để thoát ký tự ===
    return f"""
    Bạn là một chuyên gia dán nhãn dữ liệu cho công cụ tìm kiếm ẩm thực Việt Nam.
    Nhiệm vụ của bạn là đánh giá mức độ liên quan của một TÀI LIỆU đối với một TRUY VẤN của người dùng.

    Hãy sử dụng thang điểm sau:
    - 3: Rất liên quan (Chính xác là thứ người dùng muốn tìm. Ví dụ: query "cơm chay" -> doc "Quán cơm chay An Lạc").
    - 2: Khá liên quan (Liên quan đến chủ đề, nhưng không phải câu trả lời trực tiếp. Ví dụ: query "cơm chay" -> doc "Bán đồ khô, thực phẩm chay").
    - 1: Hơi liên quan (Chỉ nhắc đến từ khóa nhưng sai ngữ cảnh. Ví dụ: query "cơm chay" -> doc "Quán bún bò gần quán cơm chay An Lạc").
    - 0: Không liên quan (Hoàn toàn sai chủ đề).

    Hãy chỉ trả lời bằng một đối tượng JSON duy nhất có định dạng {{"label": <số_điểm>}}.
    Không thêm bất kỳ văn bản giải thích hay markdown nào.

    ---
    TRUY VẤN CỦA NGƯỜI DÙNG:
    "{query}"

    TÀI LIỆU CẦN ĐÁNH GIÁ:
    "{document_text}"
    ---

    JSON KẾT QUẢ:
    """

def get_label_from_llm(query, document_text):
    """Gửi yêu cầu đến Gemini và phân tích kết quả JSON."""
    
    prompt = create_prompt(query, document_text)
    
    try:
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned_response)
        label = int(data.get("label", -1))
        if label not in [0, 1, 2, 3]:
            return -1
        return label
        
    except json.JSONDecodeError:
        print(f"Lỗi JSONDecodeError: Không thể phân tích response: {cleaned_response}")
        return -1
    except Exception as e:
        print(f"Lỗi API hoặc lỗi khác: {e}")
        return -1

# ==============================================================================
# PHẦN 3: XỬ LÝ CHÍNH
# ==============================================================================

def main():
    print(f"🚀 Bắt đầu quá trình dán nhãn tự động với {MODEL_NAME}...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy thư mục '{INPUT_DIR}'.")
        print("Vui lòng tạo thư mục và đặt các file CSV của bạn vào đó.")
        return

    if not csv_files:
        print(f"Không tìm thấy file .csv nào trong thư mục '{INPUT_DIR}'.")
        return

    print(f"Tìm thấy {len(csv_files)} file CSV để xử lý: {csv_files}\n")

    for csv_file in csv_files:
        start_time_file = time.time()
        
        # === SỬA LỖI 2: Cải thiện logic lấy query từ tên file ===
        base_name = os.path.splitext(csv_file)[0]
        if base_name.startswith("search_results_"):
            base_name = base_name[len("search_results_"):]
        query_text = base_name.replace('_', ' ')

        print(f"--- Đang xử lý file: {csv_file} (Truy vấn: '{query_text}') ---")
        
        file_path = os.path.join(INPUT_DIR, csv_file)
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Lỗi khi đọc file {file_path}: {e}")
            continue

        df = df.fillna('')
        labels = []
        
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Dán nhãn '{query_text}'"):
            doc_text = (
                f"Tên: {row.get('name', '')}. "
                f"Bình luận: {row.get('comment', '')}. "
                f"Mô tả: {row.get('text_for_embedding', '')}"
            )
            
            label = get_label_from_llm(query_text, doc_text)
            labels.append(label)
            
            time.sleep(RATE_LIMIT_DELAY)

        df['llm_label'] = labels
        output_path = os.path.join(OUTPUT_DIR, f"labeled_{csv_file}")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        end_time_file = time.time()
        print(f"✅ Hoàn thành file '{csv_file}' sau {end_time_file - start_time_file:.2f} giây.")
        print(f"Kết quả đã được lưu tại: {output_path}\n")

    print("🎉 Đã hoàn thành tất cả các file!")

if __name__ == "__main__":
    main()
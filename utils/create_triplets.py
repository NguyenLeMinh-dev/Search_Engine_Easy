import pandas as pd
import numpy as np
import faiss
import random
import json
import time
import re
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import os # Dùng để lấy API key an toàn

# --- 1. CẤU HÌNH (THAY ĐỔI CÁC GIÁ TRỊ NÀY) ---

CSV_FILE_PATH = '/home/minh/Documents/SEG_project/datas/datas_crawl/final_processed_data.csv' 
ID_COLUMN_NAME = 'id'
COLUMN_TO_USE = 'text_for_embedding'
SBERT_MODEL_NAME = 'dangvantuan/vietnamese-embedding'
SBERT_MAX_LENGTH = 256
LLM_MODEL_NAME = 'gemini-2.0-flash' 

# !!! LẤY API KEY TỪ BIẾN MÔI TRƯỜNG (An toàn hơn)
# Chạy 'export GEMINI_API_KEY="KEY_CUA_BAN"' trong terminal trước
# AIzaSyADR9Duwd8NvqbLth1QHRMwFK8v0BVNzfM
GEMINI_API_KEY = 'AIzaSyDTfQ02luFUaPkhrHEYHaEgit0d2Mh53is'
if not GEMINI_API_KEY:
    print("LỖI: Không tìm thấy GEMINI_API_KEY trong biến môi trường.")
    exit()

TOP_K_HARD = 20
NUM_EASY = 15

# (Không cần MAX_WORKERS khi chạy tuần tự)

# --- 2. KHỞI TẠO MODEL GEMINI ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
    generation_config = genai.GenerationConfig(
        response_mime_type="application/json"
    )
    gemini_model = genai.GenerativeModel(
        LLM_MODEL_NAME,
        generation_config=generation_config
    )
    print(f"Đã khởi tạo model Gemini: {LLM_MODEL_NAME}")
except Exception as e:
    print(f"LỖI: Không thể khởi tạo model Gemini. Kiểm tra API Key và tên model. Lỗi: {e}")
    exit()

# --- 3. CÁC HÀM XỬ LÝ (Giữ nguyên) ---

def load_data_and_encode(csv_path, id_column, text_column, model_name, max_length):
    """
    Đọc file CSV, mã hóa văn bản và trả về dữ liệu.
    *** Đã bao gồm SỬA LỖI CUDA ***
    """
    print(f"Đang tải dữ liệu từ: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if id_column not in df.columns or text_column not in df.columns:
        raise ValueError(f"Không tìm thấy cột '{id_column}' hoặc '{text_column}' trong file CSV.")
        
    df = df[[id_column, text_column]].dropna()
    df[id_column] = df[id_column].astype(str)
    
    ids = df[id_column].tolist()
    texts = df[text_column].tolist()
    
    print(f"Đã tải {len(texts)} dòng văn bản.")
    
    print(f"Đang tải mô hình SBERT: {model_name}...")
    model = SentenceTransformer(model_name)
    
    model.max_seq_length = max_length
    print(f"Đã ép model.max_seq_length = {max_length} để tránh lỗi CUDA.")
    
    print("Đang mã hóa văn bản (có thể mất vài phút)...")
    embeddings = model.encode([str(t) for t in texts], show_progress_bar=True)
    print(f"Mã hóa hoàn tất. Shape của embeddings: {embeddings.shape}")
    
    return ids, texts, embeddings

def build_faiss_index(embeddings):
    """Xây dựng chỉ mục FAISS."""
    print("Đang xây dựng chỉ mục FAISS...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    print(f"Xây dựng FAISS hoàn tất. Tổng số vector: {index.ntotal}")
    return index

def get_candidates(anchor_faiss_index, faiss_index, total_items, top_k, num_easy):
    """Lấy danh sách ứng viên Khó và Dễ."""
    anchor_vector = np.array([faiss_index.reconstruct(anchor_faiss_index)], dtype=np.float32)
    distances, hard_indices = faiss_index.search(anchor_vector, top_k + 1)
    hard_candidate_indices = hard_indices[0][1:].tolist()
    
    all_indices_set = set(range(total_items))
    all_indices_set.discard(anchor_faiss_index)
    all_indices_set.difference_update(hard_candidate_indices)
    
    easy_candidate_indices = random.sample(list(all_indices_set), min(num_easy, len(all_indices_set)))
    
    return hard_candidate_indices, easy_candidate_indices

def get_llm_classification(anchor_id, anchor_text, candidate_data):
    """Gọi Gemini để phân loại các ứng viên."""
    prompt_header = """Bạn là một chuyên gia Khoa học Dữ liệu, đang thực hiện 'triplet mining'. 
Nhiệm vụ của bạn là đọc Anchor và phân loại các Candidate thành 3 nhóm:
1.  **positive**: Cùng loại hình/món ăn chính.
2.  **hard_negative**: Dễ gây nhầm lẫn (cùng danh mục rộng nhưng khác món chính).
3.  **easy_negative**: Hoàn toàn không liên quan (ví dụ: Bò Né vs Trà Sữa).

Chỉ trả về ID của các candidate dưới dạng JSON. KHÔNG giải thích.
Schema JSON phải là: {"positive": [id1, id2], "hard_negative": [id3, id4], "easy_negative": [id5]}
"""
    anchor_prompt = f"\n--- ANCHOR ---\nID: {anchor_id}\nText: \"{anchor_text}\"\n"
    candidates_prompt = "\n--- CANDIDATES ---\n"
    for cand_id, cand_text, cand_type in candidate_data:
        candidates_prompt += f"ID: {cand_id} (Gợi ý: {cand_type})\nText: \"{cand_text}\"\n\n"
    final_prompt = prompt_header + anchor_prompt + candidates_prompt
    
    raw_response_text = ""
    try:
        completion = gemini_model.generate_content(final_prompt)
        raw_response_text = completion.text
        json_match = re.search(r"\{.*\}", raw_response_text, re.DOTALL)
        if not json_match:
            raise ValueError("Không tìm thấy khối JSON hợp lệ trong phản hồi.")
        json_string = json_match.group(0)
        result = json.loads(json_string)
        return {
            "positive": result.get("positive", []),
            "hard_negative": result.get("hard_negative", []),
            "easy_negative": result.get("easy_negative", [])
        }
    except Exception as e:
        print(f"LỖI khi gọi LLM (Gemini) cho Anchor ID {anchor_id}: {e}")
        print(f"Gemini response raw: {raw_response_text}")
        if 'completion' in locals():
            print(f"Gemini prompt feedback: {completion.prompt_feedback}")
        return {"positive": [], "hard_negative": [], "easy_negative": []}

def process_anchor(faiss_anchor_index, total_items, id_map, text_map, faiss_index):
    """Hàm này xử lý MỘT anchor duy nhất."""
    start_time = time.time()
    anchor_original_id = id_map[faiss_anchor_index]
    anchor_text = text_map[faiss_anchor_index]
    
    print(f"--- Bắt đầu Anchor {faiss_anchor_index + 1}/{total_items} (ID: {anchor_original_id}) ---")
    
    hard_indices, easy_indices = get_candidates(
        faiss_anchor_index, 
        faiss_index, 
        total_items, 
        TOP_K_HARD, 
        NUM_EASY
    )
    
    candidate_data_for_llm = []
    for idx in hard_indices:
        candidate_data_for_llm.append((id_map[idx], text_map[idx], "hard_potential"))
    for idx in easy_indices:
        candidate_data_for_llm.append((id_map[idx], text_map[idx], "easy_potential"))
        
    classification = get_llm_classification(anchor_original_id, anchor_text, candidate_data_for_llm)
    
    anchor_triplets = []
    positive_ids = classification.get('positive', [])
    hard_neg_ids = classification.get('hard_negative', [])
    easy_neg_ids = classification.get('easy_negative', [])

    for pos_id in positive_ids:
        for hard_neg_id in hard_neg_ids:
            anchor_triplets.append((anchor_original_id, pos_id, hard_neg_id))
        for easy_neg_id in easy_neg_ids:
            anchor_triplets.append((anchor_original_id, pos_id, easy_neg_id))
            
    end_time = time.time()
    print(f"--- Hoàn tất Anchor {faiss_anchor_index + 1} (ID: {anchor_original_id}) trong {end_time - start_time:.2f}s. Thêm {len(anchor_triplets)} triplet. ---")
    
    return anchor_triplets

# --- 4. HÀM CHÍNH (MAIN) - ĐÃ SỬA LẠI ĐỂ CHẠY TUẦN TỰ VÀ SLEEP ---

# --- 4. HÀM CHÍNH (MAIN) - ĐÃ SỬA LẠI ĐỂ CHẠY BATCH 200 ANCHOR ---

def main():
    # --- THÊM CẤU HÌNH BATCH ---
    START_INDEX = 500  # Bắt đầu từ anchor 0
    NUM_TO_RUN = 200  # Chạy 200 anchor
    # Lần sau chạy, bạn chỉ cần đổi START_INDEX = 200
    # Lần sau nữa, START_INDEX = 400, v.v.
    # --- KẾT THÚC CẤU HÌNH BATCH ---

    # Giai đoạn 1: Mã hóa và Xây dựng chỉ mục
    original_ids, texts, embeddings = load_data_and_encode(
        CSV_FILE_PATH, 
        ID_COLUMN_NAME,
        COLUMN_TO_USE, 
        SBERT_MODEL_NAME,
        SBERT_MAX_LENGTH
    )
    faiss_index = build_faiss_index(embeddings)
    
    total_items = len(texts)
    all_triplets = []
    
    id_map = {i: org_id for i, org_id in enumerate(original_ids)}
    text_map = {i: txt for i, txt in enumerate(texts)} 

    # Tính toán index kết thúc cho batch này
    END_INDEX = min(START_INDEX + NUM_TO_RUN, total_items)

    # Giai đoạn 2: Chạy Tuần tự (Sequential)
    print(f"--- BẮT ĐẦU CHẠY BATCH (Anchor {START_INDEX} đến {END_INDEX}) ---")
    print(f"Sẽ chờ 4.1 giây sau mỗi request (Quota 15 RPM).")
    
    # Sửa vòng lặp for để chỉ chạy trong batch
    for faiss_anchor_index in range(START_INDEX, END_INDEX):
        
        # Chạy hàm xử lý cho từng anchor
        anchor_triplets = process_anchor(
            faiss_anchor_index,
            total_items,
            id_map,
            text_map,
            faiss_index
        )
        
        # Thêm kết quả
        all_triplets.extend(anchor_triplets)
        
        # --- THÊM TIME SLEEP ---
        # 60 giây / 15 requests = 4 giây/request. Thêm 0.1s cho an toàn.
        time.sleep(4.1) 

    # Giai đoạn 3: Lưu kết quả
    print(f"\n--- HOÀN TẤT BATCH {START_INDEX}_to_{END_INDEX} ---")
    print(f"Tổng cộng đã tạo được {len(all_triplets)} bộ ba trong batch này.")
    
    triplet_df = pd.DataFrame(all_triplets, columns=['anchor_id', 'positive_id', 'negative_id'])
    
    # Sửa tên file output để lưu theo batch
    output_filename = f'triplets_dataset_{START_INDEX}_to_{END_INDEX}.csv'
    triplet_df.to_csv(output_filename, index=False)
    print(f"Đã lưu kết quả vào file '{output_filename}'")

# --- 5. CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    main()

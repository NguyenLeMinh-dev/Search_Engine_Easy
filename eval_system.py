import pandas as pd
import numpy as np
import os

# ==============================================================================
# PHẦN 1: CẤU HÌNH ĐÁNH GIÁ (Cập nhật đường dẫn GT)
# ==============================================================================

# 1. Chỉ định thư mục và tệp
RESULT_DIR = "/home/minh/Documents/SEG_project/datas/queries_lable"  # Thư mục "Bài làm" (Kết quả mới)

# (THAY ĐỔI): Không còn là thư mục GT, mà là 1 tệp QREL duy nhất
GT_QRELS_FILE = "/home/minh/Documents/SEG_project/datas/all_labels.qrels.CLEANED.csv" 

# 2. Cấu hình cột
LABEL_COLUMN = "llm_label" # Tên cột label trong file QREL (phải khớp với code combine)
RELEVANCE_THRESHOLD = 2
K_VALUES = [10, 50, 100]
ID_LENGTH = 6 # Độ dài ID chuẩn (ví dụ: '001027')

# ==============================================================================
# PHẦN 2: CÁC HÀM TÍNH TOÁN (Giữ nguyên)
# ==============================================================================

def calculate_dcg(relevance_scores, k):
    """Tính Discounted Cumulative Gain (DCG) tại K."""
    scores = np.asarray(relevance_scores)[:k]
    if scores.size == 0:
        return 0.0
    discounts = np.log2(np.arange(scores.size) + 2)
    return np.sum(scores / discounts)

def calculate_ap(predicted_scores, total_relevant_count, k):
    """Tính Average Precision (AP) tại K (chia cho tổng số liên quan)"""
    if total_relevant_count == 0:
        return 0.0
    binary_relevance = [1 if score >= RELEVANCE_THRESHOLD else 0 for score in predicted_scores[:k]]
    precision_values = []
    hits = 0
    for i, rel in enumerate(binary_relevance):
        if rel == 1:
            hits += 1
            precision_at_i = hits / (i + 1)
            precision_values.append(precision_at_i)
    if not precision_values:
        return 0.0
    return np.sum(precision_values) / total_relevant_count

# ==============================================================================
# PHẦN 3: CHẠY ĐÁNH GIÁ CHÍNH (LOGIC MỚI)
# ==============================================================================

def load_ground_truth(qrels_file_path):
    """
    (LOGIC MỚI)
    Tải tệp QREL duy nhất và xây dựng một map tra cứu lồng nhau.
    Cấu trúc: { 'query_id_A': {'doc_1': 3, 'doc_2': 0}, ... }
    """
    print(f"Đang tải Ground Truth từ tệp QREL: {qrels_file_path}...")
    try:
        qrels_df = pd.read_csv(qrels_file_path, dtype={'doc_id': str})
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy tệp QREL: {qrels_file_path}")
        return None

    # (Đảm bảo tên cột khớp với tệp QREL bạn đã tạo)
    if 'query_id' not in qrels_df.columns or \
       'doc_id' not in qrels_df.columns or \
       'relevance_score' not in qrels_df.columns:
        print(f"LỖI: Tệp QREL phải chứa 3 cột: 'query_id', 'doc_id', 'relevance_score'")
        return None

    ground_truth = {}
    
    # Đổi tên cột 'relevance_score' thành LABEL_COLUMN để khớp code cũ
    qrels_df[LABEL_COLUMN] = qrels_df['relevance_score'].fillna(0).replace(-1, 0)
    qrels_df['doc_id'] = qrels_df['doc_id'].astype(str).str.zfill(ID_LENGTH)

    for _, row in qrels_df.iterrows():
        query_id = row['query_id']
        doc_id = row['doc_id']
        score = row[LABEL_COLUMN]
        
        if query_id not in ground_truth:
            ground_truth[query_id] = {}
        ground_truth[query_id][doc_id] = score
        
    print(f"Tải xong! Đã tìm thấy nhãn cho {len(ground_truth)} truy vấn.")
    return ground_truth


def main_evaluation():
    print(f"🚀 Bắt đầu đánh giá TOÀN BỘ HỆ THỐNG...")
    
    # --- 1. (LOGIC MỚI) Tải Ground Truth 1 LẦN DUY NHẤT ---
    ground_truth = load_ground_truth(GT_QRELS_FILE)
    if ground_truth is None:
        return

    # --- 2. (THAY ĐỔI) Tìm tất cả các file "Bài làm" (Result) ---
    print(f"   (Đọc 'Bài làm' từ: {RESULT_DIR}/)\n")
    try:
        result_files = [f for f in os.listdir(RESULT_DIR) if f.endswith('.csv')]
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy thư mục 'Bài làm' (Result): {RESULT_DIR}")
        return
    
    if not result_files:
        print(f"Không tìm thấy file .csv (Bài làm) nào trong {RESULT_DIR}")
        return

    print(f"Tìm thấy {len(result_files)} file 'Bài làm' để đánh giá...\n")

    # --- 3. Chuẩn bị list để lưu điểm của TẤT CẢ query ---
    all_scores = {k: {'ndcg': [], 'ap': []} for k in K_VALUES}
    evaluated_query_count = 0

    # --- 4. (THAY ĐỔI) Lặp qua từng file "Bài làm" ---
    for result_filename in result_files:
        
        # --- 4a. Xác định tên file và query ---
        result_path = os.path.join(RESULT_DIR, result_filename)
        
        # Lấy query_id từ tên file "Bài làm" (không có .csv)
        query_id_from_result = os.path.splitext(result_filename)[0]
        
        # *QUAN TRỌNG*: Xây dựng query_id trong QREL dựa trên logic của file cũ
        # File "Bài làm" (Result): search_results_com_chay.csv
        # File "Đáp án" (GT) cũ: labeled_search_results_com_chay.csv
        # ==> query_id trong QREL sẽ là: "labeled_search_results_com_chay"
        query_id_in_qrel = "labeled_" + query_id_from_result
        
        query_text = query_id_from_result.replace("search_results_", "").replace('_', ' ')

        print(f"--- Đang đánh giá Query: '{query_text}' ---")

        # --- 4b. (LOGIC MỚI) Lấy nhãn từ Map đã tải ---
        label_map = ground_truth.get(query_id_in_qrel)
        
        if not label_map:
            print(f"CẢNH BÁO: Không tìm thấy nhãn (Ground Truth) cho query_id '{query_id_in_qrel}' trong tệp QREL. Bỏ qua.")
            continue
        
        # --- 4c. Tải file "Bài làm" ---
        try:
            df_result = pd.read_csv(result_path, dtype={'id': str})
        except Exception as e:
            print(f"Lỗi đọc file 'Bài làm' {result_path}: {e}. Bỏ qua.")
            continue

        # --- 4d. Chuẩn hóa ID và tra cứu nhãn ---
        df_result['id'] = df_result['id'].astype(str).str.zfill(ID_LENGTH)
        predicted_ids = df_result['id'].tolist()
        
        predicted_relevance_scores = [label_map.get(id, 0) for id in predicted_ids]
        
        # (LOGIC MỚI) Lấy ideal scores TỪ MAP
        all_known_scores = list(label_map.values())
        ideal_relevance_scores = sorted(all_known_scores, reverse=True)
        total_relevant_docs = sum(1 for score in all_known_scores if score >= RELEVANCE_THRESHOLD)

        # --- 4e. Tính và LƯU điểm của query này (Giữ nguyên logic tính) ---
        for k in K_VALUES:
            k_val = min(k, len(predicted_relevance_scores)) 
            
            dcg_at_k = calculate_dcg(predicted_relevance_scores, k_val)
            idcg_at_k = calculate_dcg(ideal_relevance_scores, k_val)
            ndcg_at_k = dcg_at_k / idcg_at_k if idcg_at_k > 0 else 0.0
            
            ap_at_k = calculate_ap(predicted_relevance_scores, total_relevant_docs, k_val)
            
            all_scores[k]['ndcg'].append(ndcg_at_k)
            all_scores[k]['ap'].append(ap_at_k)

        # In điểm tóm tắt của query này
        k_10 = K_VALUES[0]
        k_100 = K_VALUES[-1]
        print(f"   -> nDCG@{k_10}: {all_scores[k_10]['ndcg'][-1]:.4f}, mAP@{k_100}: {all_scores[k_100]['ap'][-1]:.4f}")
        evaluated_query_count += 1

    # --- 5. Tính toán và In kết quả TRUNG BÌNH ---
    if evaluated_query_count == 0:
        print("\nKHÔNG CÓ TRUY VẤN NÀO ĐƯỢC ĐÁNH GIÁ. Vui lòng kiểm tra lại đường dẫn và tên tệp.")
        return

    print("\n" + "="*45)
    print(f"✅ ĐÁNH GIÁ HỆ THỐNG HOÀN TẤT ({evaluated_query_count} truy vấn)")
    print(f"{'K':<5} | {'Mean nDCG@k (m-nDCG)':<20} | {'Mean AP@k (mAP)':<15}")
    print("-" * 45)

    for k in K_VALUES:
        mean_ndcg = np.mean(all_scores[k]['ndcg'])
        mean_ap = np.mean(all_scores[k]['ap'])
        print(f"{k:<5} | {mean_ndcg:<20.4f} | {mean_ap:<15.4f}")
    
    print("="*45)

if __name__ == "__main__":
    main_evaluation()
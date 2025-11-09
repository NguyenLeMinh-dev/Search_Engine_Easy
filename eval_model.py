import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import sys
import os # Thêm import os

# --- 1. CẤU HÌNH ---

DATA_CSV_PATH = '/home/minh/Documents/SEG_project/datas/datas_crawl/final_processed_data.csv'
ID_COLUMN = 'id'
TEXT_COLUMN = 'text_for_embedding'
MODEL_GOC_PATH = 'dangvantuan/vietnamese-embedding'
MODEL_FINETUNE_PATH = '/home/minh/Documents/SEG_project/datas/triplets_file/fintune_sbert_v1' # Thư mục model bạn đã lưu
SBERT_MAX_LENGTH = 256
K_TOP = 10 # Đánh giá Top-10 kết quả

# --- 2. CẤU HÌNH FILE LABEL (SỬA ĐỔI) ---
# Ánh xạ từ query bạn muốn test sang tên file label tương ứng
LABEL_FILES_MAP = {
    "bánh mì": '/home/minh/Documents/SEG_project/datas/label/labeled_search_results_bánh_mì.csv',          # <<< THAY TÊN FILE LABEL CHO BÁNH MÌ
    "buffet nướng": '/home/minh/Documents/SEG_project/datas/label/labeled_search_results_buffet_nướng.csv', # <<< THAY TÊN FILE LABEL CHO BUFFET
    "cơm chay": '/home/minh/Documents/SEG_project/datas/label/labeled_search_results_cơm_chay.csv'          # <<< THAY TÊN FILE LABEL CHO CƠM CHAY
}
# Tên cột trong các file label
LABEL_ID_COLUMN = 'id'           # <<< Tên cột chứa ID item trong file label
SCORE_COLUMN = 'llm_label'       # <<< Tên cột chứa điểm (0-3) trong file label

# --- HÀM TỰ ĐỘNG TẠO TEST_QUERIES TỪ NHIỀU FILE LABEL (SỬA ĐỔI) ---
def build_test_queries_from_label_files(label_files_map, id_col, score_col):
    """Đọc nhiều file label và tạo cấu trúc TEST_QUERIES."""
    test_queries = {}
    print("Đang xây dựng bộ kiểm thử TEST_QUERIES từ nhiều file...")

    for query, label_file_path in label_files_map.items():
        print(f"\nĐang đọc file label cho query '{query}' từ: {label_file_path}")
        try:
            if not os.path.exists(label_file_path):
                print(f"LỖI: Không tìm thấy file label '{label_file_path}'. Bỏ qua query '{query}'.")
                test_queries[query] = {}
                continue
            label_df = pd.read_csv(label_file_path)
        except Exception as e:
            print(f"Lỗi khi đọc file label '{label_file_path}': {e}. Bỏ qua query '{query}'.")
            test_queries[query] = {}
            continue

        required_cols = [id_col, score_col]
        if not all(col in label_df.columns for col in required_cols):
            print(f"LỖI: File label '{label_file_path}' phải chứa các cột: {', '.join(required_cols)}. Bỏ qua query '{query}'.")
            test_queries[query] = {}
            continue

        # --- SỬA 2: Ép kiểu ID trong file label sang INT ---
        try:
             # Đọc có thể là string '001' hoặc int 1, ép sang str rồi ép sang int
             label_df[id_col] = label_df[id_col].astype(str).astype(int)
        except Exception as e:
             print(f"Lỗi khi chuyển đổi cột ID '{id_col}' sang INT trong file '{label_file_path}': {e}. Bỏ qua query '{query}'.")
             test_queries[query] = {}
             continue
        # --- KẾT THÚC SỬA 2 ---

        query_dict = label_df[label_df[score_col] > 0].set_index(id_col)[score_col].to_dict()
        test_queries[query] = query_dict

        if not query_dict:
             print(f"Cảnh báo: Không tìm thấy nhãn liên quan (điểm > 0) cho query '{query}' trong file '{label_file_path}'.")
        else:
             print(f"  -> Đã thêm query '{query}' với {len(query_dict)} nhãn liên quan (điểm > 0).")

    if all(not v for v in test_queries.values()):
        print("\nLỖI: Không xây dựng được TEST_QUERIES. Kiểm tra lại đường dẫn file label và cấu trúc file.")
        return None

    print("\nXây dựng TEST_QUERIES hoàn tất.")
    return test_queries

# --- HÀM LOAD DATA (SỬA ĐỔI) ---
def load_data_and_model(model_path):
    """Tải model và toàn bộ văn bản từ CSV."""
    print(f"\n--- Đang tải model: {model_path} ---")
    try:
        model = SentenceTransformer(model_path)
        model.max_seq_length = SBERT_MAX_LENGTH
    except Exception as e:
        print(f"Lỗi khi tải model '{model_path}': {e}")
        sys.exit(1)

    print(f"Đang tải dữ liệu từ: {DATA_CSV_PATH}")
    try:
        # Đọc ID ban đầu là string để xử lý linh hoạt
        df = pd.read_csv(DATA_CSV_PATH, dtype={ID_COLUMN: str})
        df = df[[ID_COLUMN, TEXT_COLUMN]].dropna()
        df = df.drop_duplicates(subset=[ID_COLUMN])
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file data '{DATA_CSV_PATH}'")
        sys.exit(1)
    except Exception as e:
        print(f"Lỗi khi đọc file data: {e}")
        sys.exit(1)

    # --- SỬA 1: Ép kiểu ID sang INT sau khi đọc ---
    try:
        df[ID_COLUMN] = df[ID_COLUMN].astype(int)
    except Exception as e:
        print(f"Lỗi khi chuyển đổi cột ID '{ID_COLUMN}' sang INT trong file data: {e}")
        print("Kiểm tra xem cột ID có chứa giá trị không phải số không.")
        sys.exit(1)
    # --- KẾT THÚC SỬA 1 ---

    texts = df[TEXT_COLUMN].astype(str).tolist()
    ids = df[ID_COLUMN].tolist() # ids bây giờ là list các số nguyên

    print(f"Đã tải {len(texts)} văn bản.")
    return model, texts, ids # Trả về ids là list INT

# --- Các hàm create_faiss_index, calculate_dcg, calculate_ndcg, evaluate_model giữ nguyên ---
def create_faiss_index(model, texts):
    """Mã hóa tất cả văn bản và tạo chỉ mục FAISS."""
    print(f"Bắt đầu mã hóa {len(texts)} văn bản...")
    try:
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    except Exception as e:
        print(f"Lỗi khi mã hóa văn bản: {e}")
        sys.exit(1)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension) # Dùng L2 distance
    index.add(np.array(embeddings, dtype=np.float32))
    print(f"Tạo FAISS Index hoàn tất. Tổng số vector: {index.ntotal}")
    return index

def calculate_dcg(relevance_scores):
    """Tính toán Discounted Cumulative Gain (DCG)."""
    scores = np.asarray(relevance_scores)
    discounts = np.log2(np.arange(2, scores.size + 2))
    discounts[discounts == 0] = 1e-10
    valid_indices = discounts != 0
    if not np.any(valid_indices):
         return 0.0
    return np.sum(scores[valid_indices] / discounts[valid_indices])


def calculate_ndcg(retrieved_scores, ground_truth_scores, k):
    """Tính toán Normalized DCG (NDCG@k)."""
    dcg_k = calculate_dcg(retrieved_scores[:k])
    ideal_scores = sorted(ground_truth_scores, reverse=True)
    idcg_k = calculate_dcg(ideal_scores[:k])
    if idcg_k == 0:
        return 0.0
    return dcg_k / idcg_k

def evaluate_model(model, index, item_ids_map, test_queries):
    """Chạy các query và tính toán NDCG@K."""
    print(f"\n--- Bắt đầu đánh giá model (với NDCG@{K_TOP}) ---")

    total_ndcg = 0
    query_count = 0

    for query_text, ground_truth_map in test_queries.items():
        if not ground_truth_map:
             print(f"\nCảnh báo: Bỏ qua query '{query_text}' vì không có nhãn ground truth được tìm thấy.")
             continue
        query_count += 1

        query_vector = model.encode([query_text])
        query_vector = np.array(query_vector, dtype=np.float32)

        try:
            distances, indices = index.search(query_vector, K_TOP)
        except Exception as e:
            print(f"Lỗi khi tìm kiếm FAISS cho query '{query_text}': {e}")
            continue

        result_indices = indices[0]
        # item_ids_map bây giờ map int -> int
        result_ids = [item_ids_map[i] for i in result_indices if i >= 0 and i < len(item_ids_map)]

        # ground_truth_map bây giờ có key là int
        retrieved_scores = [ground_truth_map.get(res_id, 0) for res_id in result_ids]
        retrieved_scores.extend([0] * (K_TOP - len(retrieved_scores)))

        ground_truth_scores = list(ground_truth_map.values())
        ndcg_k = calculate_ndcg(retrieved_scores, ground_truth_scores, K_TOP)

        total_ndcg += ndcg_k

        print(f"\nQuery: '{query_text}'")
        print(f"  Kết quả Top-{K_TOP} (ID): {result_ids}")
        print(f"  Điểm số (Relevance):    {retrieved_scores[:len(result_ids)]}")
        print(f"  NDCG@{K_TOP}:             {ndcg_k:.4f}")

    if query_count == 0:
        print("\nLỖI: Không có query nào hợp lệ để đánh giá.")
        return 0.0

    mean_ndcg = total_ndcg / query_count

    print("\n--- KẾT QUẢ TRUNG BÌNH ---")
    print(f"  Mean NDCG@{K_TOP}: {mean_ndcg:.4f} (trên {query_count} queries)")
    return mean_ndcg

def main():
    # Tự động tạo TEST_QUERIES từ nhiều file label
    TEST_QUERIES = build_test_queries_from_label_files(
        LABEL_FILES_MAP,
        LABEL_ID_COLUMN,
        SCORE_COLUMN
    )
    if TEST_QUERIES is None:
        sys.exit(1)

    # 1. Tải dữ liệu (chỉ cần làm 1 lần)
    _, texts, ids = load_data_and_model(MODEL_GOC_PATH)
    # Tạo map: index trong FAISS (int) -> ID gốc (INT)
    item_ids_map = {i: int_id for i, int_id in enumerate(ids)}

    # 2. Đánh giá Model Gốc
    model_goc, _, _ = load_data_and_model(MODEL_GOC_PATH)
    index_goc = create_faiss_index(model_goc, texts)
    ndcg_goc = evaluate_model(model_goc, index_goc, item_ids_map, TEST_QUERIES)

    # 3. Đánh giá Model Fine-tune
    try:
        model_finetune, _, _ = load_data_and_model(MODEL_FINETUNE_PATH)
    except Exception as e:
        print(f"\nLỖI: Không thể tải model fine-tuned từ '{MODEL_FINETUNE_PATH}'.")
        print("Hãy đảm bảo bạn đã chạy 'train_finetune.py' thành công và thư mục tồn tại.")
        print(f"Chi tiết lỗi: {e}")
        ndcg_finetune = 0.0
    else:
        index_finetune = create_faiss_index(model_finetune, texts)
        ndcg_finetune = evaluate_model(model_finetune, index_finetune, item_ids_map, TEST_QUERIES)

    # 4. In so sánh cuối cùng
    print("\n\n" + "="*30)
    print("  SO SÁNH KẾT QUẢ CUỐI CÙNG")
    print("="*30)
    print(f"Model Gốc ({MODEL_GOC_PATH}):")
    print(f"  Mean NDCG@{K_TOP}: {ndcg_goc:.4f}")
    print("\n" + "-"*30)
    print(f"Model Fine-tuned ({MODEL_FINETUNE_PATH}):")
    print(f"  Mean NDCG@{K_TOP}: {ndcg_finetune:.4f}")
    print("="*30)

    if ndcg_finetune > ndcg_goc:
        print("\nCHÚC MỪNG! Model fine-tuned 'xịn' hơn model gốc (dựa trên NDCG).")
    elif ndcg_finetune == ndcg_goc and ndcg_finetune > 0:
         print("\nHai model cho kết quả tương đương.")
    else:
        print("\nModel fine-tuned chưa tốt hơn model gốc. Có thể cần thêm dữ liệu/epochs hoặc kiểm tra lại file label.")

if __name__ == "__main__":
    main()


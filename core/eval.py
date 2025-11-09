import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import faiss
from tqdm import tqdm
import sys

# --- Import các lớp Model từ file huấn luyện ---
# Giả sử file huấn luyện của bạn tên là 'train_two_tower_pytorch_v2.py'
# Nếu tên file khác, hãy đổi tên import dưới đây
try:
    from train import ItemTower, UserTower, load_all_data # Tái sử dụng hàm load_data
    # Thêm dòng này vào gần các import khác
    from configs.config import MAX_HISTORY_LENGTH
except ImportError:
    print("LỖI: Không thể import ItemTower, UserTower từ 'train_two_tower_pytorch_v2.py'.")
    print("Hãy đảm bảo file huấn luyện nằm cùng thư mục hoặc trong PYTHONPATH.")
    sys.exit(1)
except Exception as e:
    print(f"Lỗi khi import từ file huấn luyện: {e}")
    sys.exit(1)


# --- 1. CẤU HÌNH ---

# Đường dẫn đến dữ liệu item và các file map ID (từ thư mục model tốt nhất)
PROCESSED_ITEM_DATA_DIR = '/home/minh/Documents/SEG_project/core/data/processed_item_data' # Vẫn cần để load item features ban đầu
BEST_MODEL_DIR = '/home/minh/Documents/SEG_project/core/models_v3_finetune'
MAPPINGS_DIR = os.path.join(BEST_MODEL_DIR, 'mappings')

# Đường dẫn đến file Test Set
INTERACTION_SPLITS_DIR = '/home/minh/Documents/SEG_project/core/data/interaction_splits'
TEST_INTERACTIONS_FILE = os.path.join(INTERACTION_SPLITS_DIR, 'test_interactions.csv')

# K cho đánh giá (có thể dùng nhiều giá trị)
EVAL_K_VALUES = [10, 50, 100]

# Hyperparameters (phải khớp với lúc huấn luyện)
EMBEDDING_DIMENSION = 128

# Thiết bị (GPU nếu có)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 2. HÀM TẢI MODEL VÀ DỮ LIỆU TEST ---

def load_models_and_test_data():
    """Tải model tốt nhất, dữ liệu item, map ID và dữ liệu test."""
    print("--- Đang tải models, dữ liệu và mappings ---")

    # --- Tải dữ liệu item và vocabs (dùng lại hàm từ file train) ---
    # Chỉ cần lấy item_data, num_users, num_items, maps ID, vocabs
    # train_df, val_df không cần thiết ở đây
    try:
        _, _, item_data, num_users, num_items, user_id_to_idx, item_id_to_idx, item_ids_list_str, vocabs, user_history_map = load_all_data()
    except Exception as e:
         print(f"Lỗi khi tải dữ liệu ban đầu: {e}")
         print("Đảm bảo hàm load_all_data hoạt động và các file dữ liệu tồn tại.")
         sys.exit(1)


    # --- Tải Models đã lưu ---
    print(f"\nĐang tải models tốt nhất từ: {BEST_MODEL_DIR}/")
    try:
        # Khởi tạo kiến trúc model (phải giống hệt lúc train)
        user_tower = UserTower(num_users, num_items, EMBEDDING_DIMENSION)
        item_tower = ItemTower(num_items, item_data, vocabs, EMBEDDING_DIMENSION) # Truyền item_data vào đây

        # Nạp state_dict
        user_tower.load_state_dict(torch.load(os.path.join(BEST_MODEL_DIR, 'user_tower.pth'), map_location=DEVICE))
        item_tower.load_state_dict(torch.load(os.path.join(BEST_MODEL_DIR, 'item_tower.pth'), map_location=DEVICE))

        # Chuyển model lên device và đặt chế độ eval()
        user_tower = user_tower.to(DEVICE).eval()
        item_tower = item_tower.to(DEVICE).eval()
        # Chuyển các tensor index trong item_tower lên device
        if hasattr(item_tower, 'district_indices') and item_tower.district_indices is not None:
             item_tower.district_indices = item_tower.district_indices.to(DEVICE)
        if hasattr(item_tower, 'tag_indices') and item_tower.tag_indices is not None:
             item_tower.tag_indices = item_tower.tag_indices.to(DEVICE)

        print("  -> Tải models thành công.")
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file model '.pth' trong '{BEST_MODEL_DIR}'.")
        print("Hãy đảm bảo bạn đã huấn luyện thành công và model đã được lưu.")
        sys.exit(1)
    except Exception as e:
        print(f"Lỗi khi tải models: {e}")
        sys.exit(1)


    # --- Tải Dữ liệu Test ---
    print(f"\nĐang tải dữ liệu test từ: {TEST_INTERACTIONS_FILE}")
    try:
        test_df = pd.read_csv(TEST_INTERACTIONS_FILE)
        test_df = test_df[['user_id', 'item_id']].astype(str)

        # Map ID sang index
        test_df['user_idx'] = test_df['user_id'].map(user_id_to_idx)
        test_df['item_idx'] = test_df['item_id'].map(item_id_to_idx)

        # Lọc bỏ user/item không có trong tập train (không thể đánh giá)
        initial_count = len(test_df)
        test_df = test_df.dropna(subset=['user_idx', 'item_idx'])
        test_df['item_idx'] = test_df['item_idx'].astype(int)
        filtered_count = len(test_df)
        if initial_count > filtered_count:
            print(f"  -> Cảnh báo: Đã lọc bỏ {initial_count - filtered_count} tương tác test do user/item không có trong tập train.")
        print(f"  -> Đã tải {filtered_count} tương tác test.")

        # Tạo ground truth: Dict {user_idx: set(item_idx)}
        test_ground_truth = test_df.groupby('user_idx')['item_idx'].agg(set).to_dict()

    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file test: {TEST_INTERACTIONS_FILE}")
        sys.exit(1)
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu test: {e}")
        sys.exit(1)

    return user_tower, item_tower, test_ground_truth, num_items, item_id_to_idx, user_history_map

# --- 3. HÀM TÍNH TOÁN METRICS (Recall, Precision, NDCG) ---

def calculate_metrics(recommendations, ground_truth, k_values):
    """Tính Recall@k, Precision@k, NDCG@k cho một user."""
    metrics = {}
    relevant_items = ground_truth
    num_relevant = len(relevant_items)
    if num_relevant == 0:
        return {f'Recall@{k}': 0.0 for k in k_values}, \
               {f'Precision@{k}': 0.0 for k in k_values}, \
               {f'NDCG@{k}': 0.0 for k in k_values}

    max_k = max(k_values)
    top_k_recs = recommendations[:max_k]

    hits_at_k = {}
    dcg_at_k = {}
    idcg_at_k = {} # Ideal DCG

    # Tính hits và DCG tăng dần
    hits_count = 0
    current_dcg = 0.0
    ideal_dcg_list = [] # Lưu DCG lý tưởng

    for i, item_idx in enumerate(top_k_recs):
        rank = i + 1
        is_relevant = 1 if item_idx in relevant_items else 0

        # Tính hits
        if is_relevant:
            hits_count += 1

        # Tính DCG
        current_dcg += (is_relevant / np.log2(rank + 1))

        # Tính Ideal DCG (giả sử tất cả relevant item xếp đầu)
        if rank <= num_relevant:
             ideal_dcg_list.append(1 / np.log2(rank + 1))

        # Lưu lại tại các mốc K
        if rank in k_values:
            hits_at_k[rank] = hits_count
            dcg_at_k[rank] = current_dcg
            idcg_at_k[rank] = np.sum(ideal_dcg_list[:rank]) # Tính tổng DCG lý tưởng đến K

    recalls = {f'Recall@{k}': hits_at_k.get(k, hits_count) / num_relevant for k in k_values}
    precisions = {f'Precision@{k}': hits_at_k.get(k, hits_count) / k for k in k_values}
    ndcgs = {f'NDCG@{k}': (dcg_at_k.get(k, current_dcg) / idcg_at_k[k]) if idcg_at_k.get(k, 0) > 0 else 0.0 for k in k_values}


    return recalls, precisions, ndcgs

# --- 4. HÀM ĐÁNH GIÁ CHÍNH ---
def evaluate_on_test_set(user_tower, item_tower, test_ground_truth, num_items, k_values, device, user_history_map):
    """Chạy đánh giá trên toàn bộ test set."""
    user_tower.eval()
    item_tower.eval()

    # 1. Tạo embedding cho tất cả items
    print("\n--- Bắt đầu đánh giá trên Test Set ---")
    print("Đang tạo embeddings cho tất cả items...")
    all_item_indices = torch.arange(num_items, dtype=torch.long).to(device)
    with torch.no_grad():
        all_item_embeddings = item_tower(all_item_indices)

    # 2. Xây dựng Index FAISS
    print("Đang xây dựng FAISS index...")
    index = faiss.IndexFlatIP(all_item_embeddings.shape[1])
    # if device == torch.device("cuda"):
    #     res = faiss.StandardGpuResources()
    #     index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(all_item_embeddings.cpu().numpy())
    print("FAISS index sẵn sàng.")

    # 3. Lấy danh sách user cần đánh giá
    test_user_indices = list(test_ground_truth.keys())
    test_user_tensor = torch.tensor(test_user_indices, dtype=torch.long).to(device)

    # 4. Tạo user embeddings
    print("Đang tạo user embeddings cho test users...")

    # --- THAY ĐỔI: Chuẩn bị history tensor ---
    processed_histories = []
    # Lặp qua các user_idx GỐC (int)
    for user_idx in test_user_indices: 
        # Lấy lịch sử, .get(user_idx, []) trả về list rỗng nếu không có
        history = user_history_map.get(user_idx, [])

        # Truncate (lấy MAX_HISTORY_LENGTH item cuối)
        history = history[-MAX_HISTORY_LENGTH:]

        # Pad (thêm 0 vào ĐẦU)
        padding_length = MAX_HISTORY_LENGTH - len(history)
        padded_history = [0] * padding_length + history

        processed_histories.append(padded_history)

    # Chuyển list các history đã xử lý sang tensor
    test_history_tensor = torch.tensor(processed_histories, dtype=torch.long).to(device)
    # test_user_tensor đã có sẵn trên device

    with torch.no_grad():
        # Gọi model với CẢ HAI tensor
        test_user_embeddings = user_tower(test_user_tensor, test_history_tensor).cpu().numpy()

    # 5. Tìm kiếm Top K (lấy K lớn nhất)
    max_k = max(k_values)
    print(f"Đang tìm kiếm Top {max_k} recommendations...")
    _, top_k_indices_all = index.search(test_user_embeddings, max_k) # shape: [num_test_users, max_k]

    # 6. Tính toán metrics cho từng user và lấy trung bình
    print("Đang tính toán metrics...")
    all_metrics_sum = {f'{metric}@{k}': 0.0 for k in k_values for metric in ['Recall', 'Precision', 'NDCG']}
    num_test_users = len(test_user_indices)

    for i, user_idx in enumerate(tqdm(test_user_indices, desc="Evaluating Users")):
        ground_truth_items = test_ground_truth[user_idx]
        recommended_item_indices = list(top_k_indices_all[i])
        # Bỏ qua index -1 nếu có
        recommended_item_indices = [idx for idx in recommended_item_indices if idx !=-1]

        recalls, precisions, ndcgs = calculate_metrics(recommended_item_indices, ground_truth_items, k_values)

        for k in k_values:
            all_metrics_sum[f'Recall@{k}'] += recalls[f'Recall@{k}']
            all_metrics_sum[f'Precision@{k}'] += precisions[f'Precision@{k}']
            all_metrics_sum[f'NDCG@{k}'] += ndcgs[f'NDCG@{k}']

    # Tính trung bình
    avg_metrics = {metric_name: total_score / num_test_users for metric_name, total_score in all_metrics_sum.items()}

    return avg_metrics


# --- 5. HÀM CHÍNH (MAIN) ---
if __name__ == "__main__":
    # Tải models và dữ liệu test
    
    user_tower, item_tower, test_ground_truth, num_items,_,  user_history_map = load_models_and_test_data()

    # Chạy đánh giá
    start_eval_time = time.time()
    avg_metrics = evaluate_on_test_set(
        user_tower,
        item_tower,
        test_ground_truth,
        num_items,
        EVAL_K_VALUES,
        DEVICE,
        user_history_map
    )
    end_eval_time = time.time()

    # In kết quả
    print("\n\n" + "="*40)
    print(f"✅ ĐÁNH GIÁ TRÊN TEST SET HOÀN TẤT")
    print(f"   (Thời gian: {end_eval_time - start_eval_time:.2f}s)")
    print("="*40)
    print(f"{'K':<5} | {'Mean Recall@k':<15} | {'Mean Precision@k':<18} | {'Mean NDCG@k':<15}")
    print("-" * (5 + 15 + 18 + 15 + 9)) # 9 = số dấu | và khoảng trắng

    for k in EVAL_K_VALUES:
        recall = avg_metrics.get(f'Recall@{k}', 0.0)
        precision = avg_metrics.get(f'Precision@{k}', 0.0)
        ndcg = avg_metrics.get(f'NDCG@{k}', 0.0)
        print(f"{k:<5} | {recall:<15.4f} | {precision:<18.4f} | {ndcg:<15.4f}")
    print("="*40)

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
import faiss
from tqdm import tqdm
from configs.config import * # Import cấu hình

# --- 2. HÀM TẢI DỮ LIỆU ---

def load_all_data():
    """Tải dữ liệu item, tương tác train/val, và xây dựng user history map."""
    print("--- Đang tải dữ liệu ---")
    item_data = {}
    vocabs = {}
    item_ids_list_str = []
    item_id_to_idx = {}
    num_items = 0

    # --- Tải Dữ liệu Item ---
    print(f"Đang tải dữ liệu item từ: {PROCESSED_ITEM_DATA_DIR}/")
    try:
        # (Giữ nguyên phần tải item_data)
        item_ids_df = pd.read_csv(os.path.join(PROCESSED_ITEM_DATA_DIR, 'item_ids.csv'))
        item_ids_list_str = item_ids_df['id'].astype(str).tolist()
        item_id_to_idx = {item_id: i for i, item_id in enumerate(item_ids_list_str)}
        num_items = len(item_ids_list_str)
        print(f"  -> Đã tải {num_items} item IDs.")

        sbert_path = os.path.join(PROCESSED_ITEM_DATA_DIR, 'item_sbert_embeddings.npy')
        item_data['sbert'] = torch.tensor(np.load(sbert_path), dtype=torch.float32)
        print(f"  -> Đã tải SBERT embeddings ({item_data['sbert'].shape})")

        numerical_path = os.path.join(PROCESSED_ITEM_DATA_DIR, 'item_numerical_features.npy')
        item_data['numerical'] = torch.tensor(np.load(numerical_path), dtype=torch.float32)
        print(f"  -> Đã tải Numerical features ({item_data['numerical'].shape})")

        for fname in ['city_vocab.json', 'district_vocab.json', 'tag_vocab.json']:
            fpath = os.path.join(PROCESSED_ITEM_DATA_DIR, fname)
            if os.path.exists(fpath):
                with open(fpath, 'r', encoding='utf-8') as f:
                    name = fname.replace('_vocab.json', '')
                    vocabs[name] = json.load(f)
                    print(f"  -> Đã tải Vocab '{name}' ({len(vocabs[name])} mục)")
            else: print(f"Cảnh báo: Không tìm thấy file vocab: {fname}")

        cat_path = os.path.join(PROCESSED_ITEM_DATA_DIR, 'item_categorical_indices.csv')
        if os.path.exists(cat_path):
             cat_df = pd.read_csv(cat_path)
             cat_df['id'] = cat_df['id'].astype(str)
             cat_df = cat_df.set_index('id').loc[item_ids_list_str].reset_index()
             if 'city_index' in cat_df.columns: item_data['city_idx'] = torch.tensor(cat_df['city_index'].values, dtype=torch.long)
             if 'district_index' in cat_df.columns: item_data['district_idx'] = torch.tensor(cat_df['district_index'].values, dtype=torch.long)
             if 'tag_indices' in cat_df.columns:
                 tag_lists = cat_df['tag_indices'].apply(json.loads).tolist()
                 max_tags = max(len(tags) for tags in tag_lists if tags) if tag_lists else 0
                 padded_tags = [tags + [0] * (max_tags - len(tags)) for tags in tag_lists]
                 item_data['tag_indices'] = torch.tensor(padded_tags, dtype=torch.long)
                 print(f"  -> Đã tải và padding Tag indices (shape: {item_data['tag_indices'].shape})")
             print(f"  -> Đã tải Categorical indices")
        else: print(f"Cảnh báo: Không tìm thấy file categorical: {cat_path}")

    except FileNotFoundError as e:
        print(f"LỖI: Thiếu file dữ liệu item trong {PROCESSED_ITEM_DATA_DIR}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu item: {e}")
        sys.exit(1)

    # --- Tải Dữ liệu User (Train & Val) ---
    print(f"\nĐang tải dữ liệu tương tác user từ: {INTERACTION_SPLITS_DIR}/")
    try:
        train_df = pd.read_csv(TRAIN_INTERACTIONS_FILE)
        val_df = pd.read_csv(VAL_INTERACTIONS_FILE)

        unique_user_ids_train = train_df['user_id'].astype(str).unique()
        user_id_to_idx = {user_id: i for i, user_id in enumerate(unique_user_ids_train)}
        num_users = len(user_id_to_idx)
        print(f"  -> Đã tải {len(train_df)} tương tác train của {num_users} users.")
        print(f"  -> Đã tải {len(val_df)} tương tác validation.")

        train_df['user_idx'] = train_df['user_id'].astype(str).map(user_id_to_idx)
        train_df['item_idx'] = train_df['item_id'].astype(str).map(item_id_to_idx)
        val_df['user_idx'] = val_df['user_id'].astype(str).map(user_id_to_idx)
        val_df['item_idx'] = val_df['item_id'].astype(str).map(item_id_to_idx)

        train_df = train_df.dropna(subset=['user_idx', 'item_idx'])
        val_df = val_df.dropna(subset=['user_idx', 'item_idx'])
        train_df['item_idx'] = train_df['item_idx'].astype(int)
        val_df['item_idx'] = val_df['item_idx'].astype(int)
        train_df['user_idx'] = train_df['user_idx'].astype(int)
        val_df['user_idx'] = val_df['user_idx'].astype(int)
        print(f"  -> Sau khi lọc user/item không xác định: {len(train_df)} train, {len(val_df)} val.")

        # --- THAY ĐỔI: XÂY DỰNG USER HISTORY MAP ---
        # Xây dựng map lịch sử user CHỈ TỪ DỮ LIỆU TRAIN
        print("  -> Đang xây dựng bản đồ lịch sử user (user history map)...")
        # Giả sử train_df chưa được sắp xếp theo thời gian, 
        # chúng ta chỉ lấy tất cả tương tác làm lịch sử
        # Sắp xếp theo user_idx để groupby hiệu quả
        train_df_sorted = train_df.sort_values(by='user_idx')
        user_history_map = train_df_sorted.groupby('user_idx')['item_idx'].apply(list).to_dict()
        print(f"  -> Đã tạo lịch sử cho {len(user_history_map)} users.")
        # ---------------------------------------------

    except FileNotFoundError as e:
        print(f"LỖI: Không tìm thấy file tương tác user trong {INTERACTION_SPLITS_DIR}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu user: {e}")
        sys.exit(1)

    # THAY ĐỔI: Trả về thêm user_history_map
    return train_df, val_df, item_data, num_users, num_items, user_id_to_idx, item_id_to_idx, item_ids_list_str, vocabs, user_history_map


# --- 5. HÀM LOSS ---
# (Giữ nguyên info_nce_loss)
def info_nce_loss(user_emb, pos_item_emb, neg_item_emb, temperature=0.1):
    pos_scores = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_scores = torch.mul(user_emb.unsqueeze(1), neg_item_emb).sum(dim=2)
    logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(user_emb.device)
    loss = nn.CrossEntropyLoss()(logits / temperature, labels)
    return loss

def info_nce_loss_in_batch(user_emb, pos_item_emb, temperature=0.1):
    """
    Tính InfoNCE Loss sử dụng các mẫu trong cùng batch làm negative.
    user_emb: [batch_size, emb_dim]
    pos_item_emb: [batch_size, emb_dim]
    """
    # [batch_size, emb_dim] x [emb_dim, batch_size] -> [batch_size, batch_size]
    # Chuẩn hóa embedding (quan trọng với cosine similarity/dot product)
    user_emb_norm = nn.functional.normalize(user_emb, p=2, dim=1)
    pos_item_emb_norm = nn.functional.normalize(pos_item_emb, p=2, dim=1)
    
    scores = torch.matmul(user_emb_norm, pos_item_emb_norm.T) / temperature
    
    # scores[i, j] là độ tương đồng của user[i] và item[j]
    # Đường chéo (i == j) là các cặp positive
    # Các phần tử còn lại (i != j) là các cặp negative
    
    # Label là [0, 1, 2, ..., batch_size-1]
    # Vì cho user[i], item positive của nó nằm ở cột [i]
    batch_size = user_emb.shape[0]
    labels = torch.arange(batch_size, dtype=torch.long).to(user_emb.device)
    
    return nn.CrossEntropyLoss()(scores, labels)

# --- 6. HÀM ĐÁNH GIÁ ---
# THAY ĐỔI: Hàm đánh giá cần nhận val_dataset để lấy được history
def evaluate_model(user_tower, item_tower, val_dataset, all_item_indices_tensor, k, device):
    """Đánh giá model trên validation set bằng Recall@K."""
    user_tower.eval() # Chuyển sang chế độ đánh giá
    item_tower.eval()
    all_recalls = []

    # 1. Tạo embedding cho TẤT CẢ items (chỉ làm 1 lần)
    print("\nĐang tạo embeddings cho tất cả items để đánh giá...")
    with torch.no_grad():
        all_item_embeddings = item_tower(all_item_indices_tensor.to(device))

    # 2. Xây dựng Index FAISS
    print("Đang xây dựng FAISS index cho items...")
    index = faiss.IndexFlatIP(all_item_embeddings.shape[1]) 
    index.add(all_item_embeddings.cpu().numpy())
    print("FAISS index sẵn sàng.")

    # 3. Lặp qua validation users
    print(f"Đang đánh giá Recall@{k} trên validation set...")
    with torch.no_grad():
        # --- THAY ĐỔI: Xử lý user và history từ val_dataset ---
        user_to_relevant_items = {}
        user_to_history = {} # Lưu lại history của mỗi user

        # Duyệt qua val_dataset để lấy user, item dương, và history
        for idx in range(len(val_dataset)):
            user_idx, item_idx, history_tensor = val_dataset[idx]
            
            # Chuyển user_idx từ numpy (nếu có) sang int
            if not isinstance(user_idx, int): user_idx = user_idx.item()
                
            if user_idx not in user_to_relevant_items:
                 user_to_relevant_items[user_idx] = set()
            user_to_relevant_items[user_idx].add(item_idx)
            
            # Chỉ cần lưu history 1 lần cho mỗi user
            if user_idx not in user_to_history:
                user_to_history[user_idx] = history_tensor

        unique_val_users = list(user_to_relevant_items.keys())
        val_user_tensor = torch.tensor(unique_val_users, dtype=torch.long).to(device)
        
        # Stack tất cả history tensor lại
        val_history_tensor = torch.stack(
            [user_to_history[u] for u in unique_val_users]
        ).to(device)
        
        # Tạo user embeddings (V3)
        val_user_embeddings = user_tower(val_user_tensor, val_history_tensor).cpu().numpy()
        # --------------------------------------------------------

        # Tìm kiếm Top K
        _, top_k_indices = index.search(val_user_embeddings, k) # shape: [num_val_users, k]

        # 4. Tính Recall cho từng user (Giữ nguyên)
        for i, user_idx in enumerate(unique_val_users):
            relevant_items = user_to_relevant_items[user_idx]
            recommended_items = set(top_k_indices[i])
            recommended_items.discard(-1) 

            hits = len(relevant_items.intersection(recommended_items))
            recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0.0
            all_recalls.append(recall)

    mean_recall = np.mean(all_recalls) if all_recalls else 0.0
    return mean_recall

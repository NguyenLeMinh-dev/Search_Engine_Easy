import os
import sys
import time
import json
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

# Import từ các file đã tách
from configs.config import *
# --- THAY ĐỔI: Import Dataset mới ---
from configs.dataset import UserItemInteractionDataset
# --- THAY ĐỔI: Import Model V3 ---
from models.user_tower_v3 import ItemTower, UserTowerV3_GRU as UserTower
# --- THAY ĐỔI: Import hàm loss IN-BATCH ---
from configs.utils import load_all_data, info_nce_loss_in_batch, evaluate_model

# --- HÀM HUẤN LUYỆN 1 EPOCH (SỬA LỖI) ---
def train_one_epoch(user_tower, item_tower, dataloader, optimizer, num_items, num_neg_samples, device):
    """Chạy 1 epoch huấn luyện (dùng In-Batch Negatives)."""
    user_tower.train() # Chuyển sang chế độ huấn luyện
    item_tower.train()
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False, unit="batch")
    
    for user_indices, pos_item_indices, history_indices in progress_bar:
        # Chuyển data lên device (GPU/CPU)
        user_indices = user_indices.to(device)
        pos_item_indices = pos_item_indices.to(device)
        history_indices = history_indices.to(device) # Thêm history
        batch_size = user_indices.size(0)

        # Tính embeddings
        user_emb = user_tower(user_indices, history_indices)
        pos_item_emb = item_tower(pos_item_indices)

        # --- SỬA LỖI: Bỏ Lấy mẫu âm (negative sampling) ngẫu nhiên ---
        # (Các dòng sau đã bị xóa)
        # neg_item_indices_flat = torch.randint(0, num_items, (batch_size * num_neg_samples,), device=device)
        # neg_item_emb = item_tower(neg_item_indices_flat)
        # neg_item_emb = neg_item_emb.view(batch_size, num_neg_samples, -1)

        # --- SỬA LỖI: Tính loss "In-Batch" ---
        loss = info_nce_loss_in_batch(user_emb, pos_item_emb, temperature=TEMPERATURE)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    return total_loss / len(dataloader)

# --- HÀM CHÍNH (MAIN) ---
def main():
    # --- THAY ĐỔI: Nhận thêm user_history_map ---
    train_df, val_df, item_data, num_users, num_items, user_id_to_idx, item_id_to_idx, item_ids_list_str, vocabs, user_history_map = load_all_data()

    # --- THAY ĐỔI: Tạo Datasets với history ---
    try:
        train_dataset = UserItemInteractionDataset(train_df, user_history_map, MAX_HISTORY_LENGTH)
        val_dataset = UserItemInteractionDataset(val_df, user_history_map, MAX_HISTORY_LENGTH)
        
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    except Exception as e:
        print(f"Lỗi khi tạo DataLoader (kiểm tra num_workers): {e}")
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


    # Khởi tạo models
    # --- THAY ĐỔI: Khởi tạo UserTower V3 (cần num_items) ---
    user_tower = UserTower(num_users, num_items, EMBEDDING_DIMENSION).to(DEVICE)
    
    if 'district_idx' in item_data: item_data['district_idx'] = item_data['district_idx'].to(DEVICE)
    if 'tag_indices' in item_data: item_data['tag_indices'] = item_data['tag_indices'].to(DEVICE)
    
    # Khởi tạo ItemTower (V3)
    item_tower = ItemTower(num_items, item_data, vocabs, EMBEDDING_DIMENSION).to(DEVICE)

    # Optimizer (Giữ nguyên)
    optimizer = optim.AdamW(
        list(user_tower.parameters()) + list(item_tower.parameters()),
        lr=LEARNING_RATE
    )

    # (Giữ nguyên phần lưu model)
    best_val_recall = -1.0
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    best_user_tower_path = os.path.join(BEST_MODEL_DIR, 'user_tower.pth')
    best_item_tower_path = os.path.join(BEST_MODEL_DIR, 'item_tower.pth')
    all_item_indices_tensor = torch.arange(num_items, dtype=torch.long)

    # Vòng lặp huấn luyện chính
    print(f"\n--- Bắt đầu huấn luyện {NUM_EPOCHS} epochs ---")
    start_train_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()

        # 1. Huấn luyện 1 epoch
        avg_train_loss = train_one_epoch(
            user_tower, item_tower, train_dataloader, optimizer, num_items, NUM_NEG_SAMPLES, DEVICE
        )

        # 2. Đánh giá trên validation set
        avg_val_recall = evaluate_model(
            user_tower, item_tower, val_dataset, all_item_indices_tensor, EVAL_K, DEVICE
        )

        epoch_end_time = time.time()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - "
              f"Time: {epoch_end_time - epoch_start_time:.2f}s - "
              f"Train Loss: {avg_train_loss:.4f} - "
              f"Val Recall@{EVAL_K}: {avg_val_recall:.4f}")

        # 3. Lưu model nếu có kết quả tốt hơn (Giữ nguyên)
        if avg_val_recall > best_val_recall:
            best_val_recall = avg_val_recall
            print(f"  -> New best validation recall! Saving model to {BEST_MODEL_DIR}")
            torch.save(user_tower.state_dict(), best_user_tower_path)
            torch.save(item_tower.state_dict(), best_item_tower_path)
            
            map_dir = os.path.join(BEST_MODEL_DIR, 'mappings')
            os.makedirs(map_dir, exist_ok=True)
            with open(os.path.join(map_dir, 'user_id_to_idx.json'), 'w') as f: json.dump(user_id_to_idx, f)
            with open(os.path.join(map_dir, 'item_id_to_idx.json'), 'w') as f: json.dump(item_id_to_idx, f)
            pd.DataFrame({'item_id': item_ids_list_str}).to_csv(os.path.join(map_dir,'item_ids_ordered.csv'), index_label='item_idx')


    end_train_time = time.time()
    print(f"\n--- Huấn luyện hoàn tất sau {end_train_time - start_train_time:.2f}s ---")
    print(f"Model tốt nhất (dựa trên Val Recall@{EVAL_K}) đã được lưu tại: {BEST_MODEL_DIR}")

if __name__ == "__main__":
    main()


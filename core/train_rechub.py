import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json
import random
from tqdm import tqdm
from collections import defaultdict

# --- Import từ Torch-RecHub (ĐÚNG ĐƯỜNG DẪN & TÊN CLASS) ---
try:
    from torch_rechub.basic.features import SparseFeature, DenseFeature, SequenceFeature
    from torch_rechub.utils.data import pad_sequences 
except ImportError:
    print("Lỗi import! Hãy chắc chắn bạn đã cài `pip install torch-rechub`")
    print("Nếu vẫn lỗi, kiểm tra lại cấu trúc thư viện torch_rechub.")
    exit()

from torch_rechub.models.matching import MIND

# --- Cấu hình (Lấy từ dự án của bạn) ---
ITEM_DATA_DIR = "/home/minh/Documents/SEG_project/core/data/processed_item_data"
INTERACTION_DIR = "/home/minh/Documents/SEG_project/core/data/interaction_splits"

SBERT_DIM = 768
NUMERICAL_DIM = 5 # Sẽ được cập nhật tự động
MAX_TAGS_PER_ITEM = 10
MAX_HISTORY_LEN = 20 

EMBEDDING_DIM = 128
NUM_NEG_SAMPLES = 4
BATCH_SIZE = 256
EPOCHS = 5
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. HÀM TẢI DỮ LIỆU (ĐÃ SỬA LỖI) ---

def load_all_data():
    """Tải tất cả vocabs, item features, và interactions."""
    print("Loading vocabularies...")
    # --- BỎ CITY_VOCAB ---
    # with open(f"{ITEM_DATA_DIR}/city_vocab.json", 'r') as f:
    #     city_vocab = json.load(f)
    with open(f"{ITEM_DATA_DIR}/district_vocab.json", 'r') as f:
        district_vocab = json.load(f)
    with open(f"{ITEM_DATA_DIR}/tag_vocab.json", 'r') as f:
        tag_vocab = json.load(f)

    print("Loading item features...")
    sbert_embs = np.load(f"{ITEM_DATA_DIR}/item_sbert_embeddings.npy")
    numerical_features = np.load(f"{ITEM_DATA_DIR}/item_numerical_features.npy")
    
    global NUMERICAL_DIM
    NUMERICAL_DIM = numerical_features.shape[1]
    print(f"Detected {NUMERICAL_DIM} numerical features.")

    item_cat_indices = pd.read_csv(f"{ITEM_DATA_DIR}/item_categorical_indices.csv")
    
    item_feature_map = {}
    for idx, row in tqdm(item_cat_indices.iterrows(), total=len(item_cat_indices), desc="Building item feature map"):
        try:
            tag_list = json.loads(row['tag_idxs'])
        except:
            tag_list = []
        
        item_feature_map[idx] = {
            'sbert': sbert_embs[idx],
            'numerical': numerical_features[idx],
            # --- BỎ 'city_idx': row['city_idx'] ---
            'district_idx': row['district_idx'],
            'tag_idxs': tag_list 
        }

    print("Loading interactions...")
    train_df = pd.read_csv(f"{INTERACTION_DIR}/train_interactions.csv")
    
    num_items = len(item_cat_indices) + 1
    num_users = train_df['user_id'].max() + 1
    
    vocab_sizes = {
        'user_id': num_users,
        'item_id': num_items, 
        # --- BỎ 'city' ---
        'district': len(district_vocab) + 1,
        'tag': len(tag_vocab) + 1,
    }
    
    return train_df, item_feature_map, vocab_sizes

def build_user_history_map(train_df):
    """Xây dựng map lịch sử {user_id: [item_idx1, item_idx2...]}."""
    print("Building user history map...")
    user_history_map = defaultdict(list)
    for user_id, item_id in train_df[['user_id', 'item_id']].values:
        user_history_map[user_id].append(item_id)
    
    for user_id in user_history_map:
        user_history_map[user_id] = user_history_map[user_id][-MAX_HISTORY_LEN:]
        
    return user_history_map

# --- 2. HÀM TẠO DỮ LIỆU HUẤN LUYỆN (ĐÃ SỬA LỖI) ---

def create_training_data(train_df, item_feature_map, user_history_map):
    """Tạo dict X và mảng y cho model.fit()."""
    
    print(f"Creating training data with {NUM_NEG_SAMPLES} negative samples per positive.")
    
    all_item_ids = list(item_feature_map.keys())
    
    X_features = defaultdict(list)
    y_labels = []

    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Generating samples"):
        user_id = row['user_id']
        pos_item_id = row['item_id']
        
        history_seq = user_history_map[user_id]
        if history_seq and history_seq[-1] == pos_item_id:
             history_seq = history_seq[:-1] 
        
        user_id_data = user_id
        history_data = history_seq 
        
        # --- 1. Thêm mẫu POSITIVE ---
        X_features['user_id'].append(user_id_data)
        X_features['hist_item_seq'].append(history_data)
        
        pos_features = item_feature_map[pos_item_id]
        X_features['item_id'].append(pos_item_id)
        X_features['sbert'].append(pos_features['sbert'])
        X_features['numerical'].append(pos_features['numerical'])
        # --- BỎ 'city_idx' ---
        X_features['district_idx'].append(pos_features['district_idx'])
        X_features['tag_idxs'].append(pos_features['tag_idxs']) 
        
        y_labels.append(1) 
        
        # --- 2. Thêm mẫu NEGATIVE ---
        for _ in range(NUM_NEG_SAMPLES):
            neg_item_id = random.choice(all_item_ids)
            while neg_item_id == pos_item_id:
                neg_item_id = random.choice(all_item_ids)
                
            X_features['user_id'].append(user_id_data)
            X_features['hist_item_seq'].append(history_data) 
        
            neg_features = item_feature_map[neg_item_id]
            X_features['item_id'].append(neg_item_id)
            X_features['sbert'].append(neg_features['sbert'])
            X_features['numerical'].append(neg_features['numerical'])
            # --- BỎ 'city_idx' ---
            X_features['district_idx'].append(neg_features['district_idx'])
            X_features['tag_idxs'].append(neg_features['tag_idxs'])
            
            y_labels.append(0) 

    # --- Padding (Giữ nguyên) ---
    print("Padding sequences...")
    X_features['hist_item_seq'] = pad_sequences(X_features['hist_item_seq'], maxlen=MAX_HISTORY_LEN, value=0)
    X_features['tag_idxs'] = pad_sequences(X_features['tag_idxs'], maxlen=MAX_TAGS_PER_ITEM, value=0)
    
    for key in X_features:
        if key not in ['hist_item_seq', 'tag_idxs']: 
             X_features[key] = np.array(X_features[key])
             
    y_labels = np.array(y_labels)
    
    return X_features, y_labels

# --- 3. ĐỊNH NGHĨA FEATURE COLUMNS (ĐÃ SỬA LỖI) ---

def define_feature_columns(vocab_sizes):
    """Khai báo features cho Torch-RecHub."""
    
    print("Defining feature columns using correct classes...")
    
    user_feature_columns = [
        SparseFeature(name='user_id', vocab_size=vocab_sizes['user_id'], embed_dim=EMBEDDING_DIM),
    ]

    item_feature_columns = [
        SparseFeature(name='item_id', vocab_size=vocab_sizes['item_id'], embed_dim=EMBEDDING_DIM),
        DenseFeature(name='sbert', embed_dim=SBERT_DIM), 
        DenseFeature(name='numerical', embed_dim=NUMERICAL_DIM), 
        # --- BỎ 'city_idx' ---
        SparseFeature(name='district_idx', vocab_size=vocab_sizes['district'], embed_dim=EMBEDDING_DIM // 4),
        SequenceFeature(name='tag_idxs', vocab_size=vocab_sizes['tag'], embed_dim=EMBEDDING_DIM // 2, pooling='mean')
    ]
    
    history_feature_columns = [
        SparseFeature(name='hist_item_seq', vocab_size=vocab_sizes['item_id'], embed_dim=EMBEDDING_DIM)
    ]
    
    history_feature_columns[0].shared_with = 'item_id'
    
    return user_feature_columns, item_feature_columns, history_feature_columns

# --- 4. TẠO PYTORCH DATASET (Giữ nguyên) ---
class RecDataset(Dataset):
    """Dataset tùy chỉnh để trả về (x_dict, y) cho DataLoader."""
    def __init__(self, x_dict, y):
        self.x_dict = {}
        for key, value in x_dict.items():
            if isinstance(value, np.ndarray):
                self.x_dict[key] = torch.tensor(value)
            else: 
                self.x_dict[key] = torch.tensor(np.array(value))

        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = {key: self.x_dict[key][idx] for key in self.x_dict}
        return x, self.y[idx]

# --- 5. HÀM MAIN HUẤN LUYỆN (Giữ nguyên) ---

def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Tải và chuẩn bị dữ liệu
    train_df, item_feature_map, vocab_sizes = load_all_data()
    user_history_map = build_user_history_map(train_df)
    X_dict, y = create_training_data(train_df, item_feature_map, user_history_map)
    
    # 2. Tạo Dataset và DataLoader
    print("Creating DataLoader...")
    train_dataset = RecDataset(X_dict, y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Định nghĩa kiến trúc (tách 3 phần)
    user_cols, item_cols, history_cols = define_feature_columns(vocab_sizes)

    # 4. Khởi tạo Model
    print("Initializing MIND model...")
    model = MIND(
        user_features=user_cols,
        item_features=item_cols,
        history_features=history_cols,
        max_len=MAX_HISTORY_LEN, 
        device=DEVICE,
        interest_dim=EMBEDDING_DIM, 
        num_interest=4, 
        temperature=0.1
    ).to(DEVICE)
    
    # 5. Định nghĩa Loss và Optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Vòng lặp Huấn luyện
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_samples = 0
        
        tk0 = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
        
        for batch_x, batch_y in tk0:
            batch_x = {k: v.to(DEVICE).long() if not torch.is_floating_point(v) else v.to(DEVICE).float() for k, v in batch_x.items()}
            batch_y = batch_y.to(DEVICE).float()

            optimizer.zero_grad()
            y_pred = model(batch_x)
            
            loss = loss_fn(y_pred.squeeze(), batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(batch_y)
            total_samples += len(batch_y)
            
            tk0.set_postfix(loss=total_loss / total_samples)
            
        print(f"Epoch {epoch+1}/{EPOCHS} - Average Loss: {total_loss / total_samples:.6f}")

    # 7. Lưu lại model
    print("Training finished. Saving model state...")
    torch.save(model.state_dict(), "rechub_mind_model.pth")
    print("Model saved successfully as 'rechub_mind_model.pth'!")

if __name__ == "__main__":
    main()
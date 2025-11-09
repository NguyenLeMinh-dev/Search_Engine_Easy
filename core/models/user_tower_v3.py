import torch
import torch.nn as nn
import json
from configs.config import * # Import DROPOUT_RATE, EMBEDDING_DIMENSION

# --- ItemTower (Lấy từ V2, đã có Dropout) ---
class ItemTower(nn.Module):
    def __init__(self, num_items, item_data, vocabs, embedding_dimension):
        super().__init__()
        self.embedding_dimension = embedding_dimension

        # Nhánh SBERT
        sbert_dim = item_data['sbert'].shape[1]
        self.sbert_embedding_layer = nn.Embedding.from_pretrained(item_data['sbert'], freeze=False) # Tạm thời vẫn freeze
        self.sbert_dense = nn.Linear(sbert_dim, embedding_dimension)
        self.sbert_relu = nn.ReLU()
        self.sbert_dropout = nn.Dropout(DROPOUT_RATE)

        # Nhánh Số
        numerical_dim = item_data['numerical'].shape[1]
        self.numerical_embedding_layer = nn.Embedding.from_pretrained(item_data['numerical'], freeze=False) # Tạm thời vẫn freeze
        self.numerical_dense = nn.Linear(numerical_dim, embedding_dimension // 4)
        self.numerical_relu = nn.ReLU()
        self.numerical_dropout = nn.Dropout(DROPOUT_RATE)

        # Nhánh District
        self.use_district = 'district' in vocabs and 'district_idx' in item_data
        district_dim = 0
        if self.use_district:
            district_vocab_size = len(vocabs['district'])
            district_dim = embedding_dimension // 8
            self.district_embedding_layer = nn.Embedding(district_vocab_size, district_dim)
            self.district_indices = item_data['district_idx'] 
        
        # Nhánh Tags
        self.use_tags = 'tag' in vocabs and 'tag_indices' in item_data
        tag_dim_pooled = 0
        if self.use_tags:
            tag_vocab_size = len(vocabs['tag'])
            tag_dim_single = embedding_dimension // 4
            tag_dim_pooled = tag_dim_single
            self.tag_embedding_layer = nn.Embedding(tag_vocab_size, tag_dim_single, padding_idx=0)
            self.tag_indices = item_data['tag_indices'] 

        fusion_input_dim = embedding_dimension + (embedding_dimension // 4)
        if self.use_district: fusion_input_dim += district_dim
        if self.use_tags: fusion_input_dim += tag_dim_pooled

        # Lớp Fusion
        self.fusion_dense_1 = nn.Linear(fusion_input_dim, embedding_dimension * 2)
        self.fusion_relu = nn.ReLU()
        self.fusion_dropout = nn.Dropout(DROPOUT_RATE)
        self.fusion_dense_2 = nn.Linear(embedding_dimension * 2, embedding_dimension)

    def forward(self, item_indices):
        sbert_emb = self.sbert_embedding_layer(item_indices)
        numerical_feat = self.numerical_embedding_layer(item_indices)
        
        sbert_projected = self.sbert_dropout(self.sbert_relu(self.sbert_dense(sbert_emb)))
        numerical_projected = self.numerical_dropout(self.numerical_relu(self.numerical_dense(numerical_feat)))

        cat_features = []
        if self.use_district:
            district_idxs = self.district_indices[item_indices]
            district_emb = self.district_embedding_layer(district_idxs)
            cat_features.append(district_emb)
        if self.use_tags:
            tag_idxs = self.tag_indices[item_indices]
            tag_embs = self.tag_embedding_layer(tag_idxs)
            mask = (tag_idxs != 0).unsqueeze(-1).float()
            sum_embs = (tag_embs * mask).sum(dim=1)
            count_embs = mask.sum(dim=1).clamp(min=1e-6)
            pooled_tags = sum_embs / count_embs
            cat_features.append(pooled_tags)

        all_features = [sbert_projected, numerical_projected] + cat_features
        concatenated_features = torch.cat(all_features, dim=1)
        
        x = self.fusion_dropout(self.fusion_relu(self.fusion_dense_1(concatenated_features)))
        final_embedding = self.fusion_dense_2(x)
        return final_embedding

# --- KIẾN TRÚC USERTOWER V3 (NÂNG CẤP VỚI GRU) ---
class UserTowerV3_GRU(nn.Module):
    """
    Tháp Người dùng (User Tower) V3
    Học từ User ID VÀ Lịch sử tương tác (dùng GRU)
    """
    def __init__(self, num_users, num_items, embedding_dimension):
        super().__init__()
        self.embedding_dimension = embedding_dimension

        # 1. Embedding cho User ID
        self.user_embedding = nn.Embedding(num_users, embedding_dimension)
        
        # 2. Embedding cho các Item trong Lịch sử
        self.item_history_embedding = nn.Embedding(
            num_items, 
            embedding_dimension, 
            padding_idx=0 # Index 0 là <PAD>
        )
        
        # 3. Lớp GRU để xử lý tuần tự
        self.gru = nn.GRU(
            input_size=embedding_dimension,
            hidden_size=embedding_dimension,
            batch_first=True # Quan trọng: [batch_size, seq_len, input_size]
        )

        # 4. Các lớp Fusion
        fusion_input_dim = embedding_dimension * 2 # (1 cho user_id_emb, 1 cho history_emb)
        
        self.fusion_dense_1 = nn.Linear(fusion_input_dim, embedding_dimension)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, user_indices, history_indices):
        # 1. Lấy User ID Embedding
        # shape: [batch_size, D]
        user_emb = self.user_embedding(user_indices)
        
        # 2. Lấy History Embedding
        # shape: [batch_size, history_length, D]
        history_embs = self.item_history_embedding(history_indices) 
        
        # 3. Đưa qua GRU
        # gru_out shape: [batch_size, history_length, D]
        # last_hidden_state shape: [num_layers, batch_size, D]
        gru_out, last_hidden_state = self.gru(history_embs)
        
        # Lấy hidden state cuối cùng làm đại diện cho lịch sử
        # shape: [batch_size, D]
        pooled_history_emb = last_hidden_state.squeeze(0) # Bỏ chiều num_layers
        
        # 4. Kết hợp
        combined_emb = torch.cat([user_emb, pooled_history_emb], dim=1)
        
        # 5. Đưa qua lớp Fusion
        final_user_emb = self.dropout(self.relu(self.fusion_dense_1(combined_emb)))
        
        return final_user_emb

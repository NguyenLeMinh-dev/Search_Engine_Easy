import torch
import torch.nn as nn
import json
from configs.config import * # Import DROPOUT_RATE

# ItemTower giữ nguyên (import từ model.py cũ)
# Tái định nghĩa ở đây để dễ quản lý
class ItemTower(nn.Module):
    def __init__(self, num_items, item_data, vocabs, embedding_dimension):
        super().__init__()
        self.embedding_dimension = embedding_dimension

        # Nhánh SBERT
        sbert_dim = item_data['sbert'].shape[1]
        self.sbert_embedding_layer = nn.Embedding.from_pretrained(item_data['sbert'], freeze=True)
        self.sbert_dense = nn.Linear(sbert_dim, embedding_dimension)
        self.sbert_relu = nn.ReLU()
        self.sbert_dropout = nn.Dropout(DROPOUT_RATE) # Thêm Dropout

        # Nhánh Số
        numerical_dim = item_data['numerical'].shape[1]
        self.numerical_embedding_layer = nn.Embedding.from_pretrained(item_data['numerical'], freeze=True)
        self.numerical_dense = nn.Linear(numerical_dim, embedding_dimension // 4)
        self.numerical_relu = nn.ReLU()
        self.numerical_dropout = nn.Dropout(DROPOUT_RATE) # Thêm Dropout

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
        self.fusion_dropout = nn.Dropout(DROPOUT_RATE) # Thêm Dropout
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

# --- KIẾN TRÚC USERTOWER V2 (NÂNG CẤP) ---
class UserTowerV2(nn.Module):
    """
    Tháp Người dùng (User Tower) V2
    Học từ User ID VÀ Lịch sử tương tác
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
        
        # 3. Các lớp Fusion
        fusion_input_dim = embedding_dimension * 2
        
        self.fusion_dense_1 = nn.Linear(fusion_input_dim, embedding_dimension)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE) # Thêm Dropout

    def forward(self, user_indices, history_indices):
        # 1. Lấy User ID Embedding
        user_emb = self.user_embedding(user_indices) # shape: [batch_size, D]
        
        # 2. Lấy History Embedding
        history_embs = self.item_history_embedding(history_indices) # shape: [batch_size, history_length, D]
        
        # 3. Tổng hợp (Pooling) Lịch sử (Masked Mean Pooling)
        mask = (history_indices != 0).unsqueeze(-1).float() 
        summed_embs = (history_embs * mask).sum(dim=1) 
        count_embs = mask.sum(dim=1).clamp(min=1e-6) 
        pooled_history_emb = summed_embs / count_embs # shape: [batch_size, D]
        
        # 4. Kết hợp
        combined_emb = torch.cat([user_emb, pooled_history_emb], dim=1)
        
        # 5. Đưa qua lớp Fusion
        final_user_emb = self.dropout(self.relu(self.fusion_dense_1(combined_emb)))
        
        return final_user_emb
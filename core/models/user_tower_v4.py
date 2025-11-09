import torch
import torch.nn as nn
import json
import math
from configs.config import * # Import DROPOUT_RATE

# --- ItemTower (ĐÃ UNFREEZE) ---
class ItemTower(nn.Module):
    """
    Tháp Sản phẩm (Item Tower)
    Cải tiến: Unfreeze SBERT và Numerical embeddings (freeze=False)
    """
    def __init__(self, num_items, item_data, vocabs, embedding_dimension):
        super().__init__()
        self.embedding_dimension = embedding_dimension

        # Nhánh SBERT
        sbert_dim = item_data['sbert'].shape[1]
        # --- THAY ĐỔI: "Mở băng" (Unfreeze) ---
        self.sbert_embedding_layer = nn.Embedding.from_pretrained(item_data['sbert'], freeze=False)
        self.sbert_dense = nn.Linear(sbert_dim, embedding_dimension)
        self.sbert_relu = nn.ReLU()
        self.sbert_dropout = nn.Dropout(DROPOUT_RATE) 

        # Nhánh Số
        numerical_dim = item_data['numerical'].shape[1]
        # --- THAY ĐỔI: "Mở băng" (Unfreeze) ---
        self.numerical_embedding_layer = nn.Embedding.from_pretrained(item_data['numerical'], freeze=False)
        self.numerical_dense = nn.Linear(numerical_dim, embedding_dimension // 4)
        self.numerical_relu = nn.ReLU()
        self.numerical_dropout = nn.Dropout(DROPOUT_RATE)

        # Nhánh District (Giữ nguyên)
        self.use_district = 'district' in vocabs and 'district_idx' in item_data
        district_dim = 0
        if self.use_district:
            district_vocab_size = len(vocabs['district'])
            district_dim = embedding_dimension // 8
            self.district_embedding_layer = nn.Embedding(district_vocab_size, district_dim)
            self.district_indices = item_data['district_idx'] 
        
        # Nhánh Tags (Giữ nguyên)
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

        # Lớp Fusion (Giữ nguyên)
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

# --- LỚP HELPER: POSITIONAL ENCODING ---
class PositionalEncoding(nn.Module):
    """
    Thêm thông tin vị trí vào embedding
    (Code chuẩn của Pytorch)
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2) # Chuyển sang [1, max_len, d_model] (cho batch_first)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# --- KIẾN TRÚC USERTOWER V4 (TRANSFORMER) ---
class UserTowerV4_Transformer(nn.Module):
    """
    Tháp Người dùng (User Tower) V4
    Sử dụng Transformer (Self-Attention) làm backbone
    """
    def __init__(self, num_users, num_items, embedding_dimension, n_heads, n_layers):
        super().__init__()
        self.embedding_dimension = embedding_dimension

        # 1. Embedding cho User ID (Giữ nguyên)
        self.user_embedding = nn.Embedding(num_users, embedding_dimension)
        
        # 2. Embedding cho các Item trong Lịch sử (Giữ nguyên)
        self.item_history_embedding = nn.Embedding(
            num_items, 
            embedding_dimension, 
            padding_idx=0 # Index 0 là <PAD>
        )
        
        # 3. Positional Encoding
        self.pos_encoder = PositionalEncoding(embedding_dimension, DROPOUT_RATE, MAX_HISTORY_LENGTH + 1)
        
        # 4. Transformer Encoder Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dimension,
            nhead=n_heads,
            dim_feedforward=embedding_dimension * 4, # Thường là 4*d_model
            dropout=DROPOUT_RATE,
            batch_first=True # Rất quan trọng!
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_layers
        )
        
        # 5. Các lớp Fusion (Giữ nguyên)
        fusion_input_dim = embedding_dimension * 2
        self.fusion_dense_1 = nn.Linear(fusion_input_dim, embedding_dimension)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE) 

    def forward(self, user_indices, history_indices):
        # 1. Lấy User ID Embedding
        user_emb = self.user_embedding(user_indices) # shape: [B, D]
        
        # 2. Lấy History Embedding
        history_embs = self.item_history_embedding(history_indices) # shape: [B, Seq_Len, D]
        
        # 3. Thêm Positional Encoding
        history_embs_with_pos = self.pos_encoder(history_embs)
        
        # 4. Tạo Padding Mask
        # Transformer cần mask=True ở những vị trí padding (idx=0)
        padding_mask = (history_indices == 0) # shape: [B, Seq_Len]
        
        # 5. Đưa qua Transformer Encoder
        transformer_out = self.transformer_encoder(
            history_embs_with_pos, 
            src_key_padding_mask=padding_mask
        ) # shape: [B, Seq_Len, D]
        
        # 6. Tổng hợp (Pooling) Lịch sử (Masked Mean Pooling)
        # Chúng ta cần pool output của Transformer, chứ không phải input
        mask = (history_indices != 0).unsqueeze(-1).float() # shape: [B, Seq_Len, 1]
        summed_embs = (transformer_out * mask).sum(dim=1) 
        count_embs = mask.sum(dim=1).clamp(min=1e-6) 
        pooled_history_emb = summed_embs / count_embs # shape: [B, D]
        
        # 7. Kết hợp
        combined_emb = torch.cat([user_emb, pooled_history_emb], dim=1)
        
        # 8. Đưa qua lớp Fusion
        final_user_emb = self.dropout(self.relu(self.fusion_dense_1(combined_emb)))
        
        return final_user_emb

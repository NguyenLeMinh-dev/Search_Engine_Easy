import torch
import torch.nn as nn
import json # Cần cho ItemTower

# --- 4. ĐỊNH NGHĨA KIẾN TRÚC MODEL ---
class ItemTower(nn.Module):
    """
    Tháp Sản phẩm (Item Tower)
    Kết hợp SBERT, Numerical, và Categorical features.
    """
    def __init__(self, num_items, item_data, vocabs, embedding_dimension):
        super().__init__()
        self.embedding_dimension = embedding_dimension

        # Nhánh SBERT
        sbert_dim = item_data['sbert'].shape[1]
        self.sbert_embedding_layer = nn.Embedding.from_pretrained(item_data['sbert'], freeze=True)
        self.sbert_dense = nn.Linear(sbert_dim, embedding_dimension)
        self.sbert_relu = nn.ReLU()

        # Nhánh Số
        numerical_dim = item_data['numerical'].shape[1]
        self.numerical_embedding_layer = nn.Embedding.from_pretrained(item_data['numerical'], freeze=True)
        self.numerical_dense = nn.Linear(numerical_dim, embedding_dimension // 4)
        self.numerical_relu = nn.ReLU()

        # Nhánh District
        self.use_district = 'district' in vocabs and 'district_idx' in item_data
        district_dim = 0
        if self.use_district:
            district_vocab_size = len(vocabs['district'])
            district_dim = embedding_dimension // 8
            self.district_embedding_layer = nn.Embedding(district_vocab_size, district_dim)
            self.district_indices = item_data['district_idx'] # Sẽ chuyển lên device sau

        # Nhánh Tags
        self.use_tags = 'tag' in vocabs and 'tag_indices' in item_data
        tag_dim_pooled = 0
        if self.use_tags:
            tag_vocab_size = len(vocabs['tag'])
            tag_dim_single = embedding_dimension // 4
            tag_dim_pooled = tag_dim_single # Kích thước sau khi pooling
            self.tag_embedding_layer = nn.Embedding(tag_vocab_size, tag_dim_single, padding_idx=0)
            self.tag_indices = item_data['tag_indices'] # Sẽ chuyển lên device sau

        # Tính toán kích thước đầu vào cho lớp Fusion
        fusion_input_dim = embedding_dimension + (embedding_dimension // 4)
        if self.use_district: fusion_input_dim += district_dim
        if self.use_tags: fusion_input_dim += tag_dim_pooled

        # Lớp Fusion
        self.fusion_dense_1 = nn.Linear(fusion_input_dim, embedding_dimension * 2)
        self.fusion_relu = nn.ReLU()
        self.fusion_dense_2 = nn.Linear(embedding_dimension * 2, embedding_dimension)

    def forward(self, item_indices):
        """
        Input: item_indices (Tensor các index của item, ví dụ: [0, 5, 10...])
        Output: Item Embeddings (Tensor shape [batch_size, embedding_dimension])
        """
        # Tra cứu các features đã được tính sẵn
        sbert_emb = self.sbert_embedding_layer(item_indices)
        numerical_feat = self.numerical_embedding_layer(item_indices)
        
        # Đưa qua các lớp Dense của từng nhánh
        sbert_projected = self.sbert_relu(self.sbert_dense(sbert_emb))
        numerical_projected = self.numerical_relu(self.numerical_dense(numerical_feat))

        cat_features = []
        # Xử lý nhánh District
        if self.use_district:
            district_idxs = self.district_indices[item_indices]
            district_emb = self.district_embedding_layer(district_idxs)
            cat_features.append(district_emb)
            
        # Xử lý nhánh Tags (với pooling)
        if self.use_tags:
            tag_idxs = self.tag_indices[item_indices] # shape: [batch_size, max_tags]
            tag_embs = self.tag_embedding_layer(tag_idxs) # shape: [batch_size, max_tags, tag_dim]
            
            # Masked average pooling (để bỏ qua các tag padding [idx=0])
            mask = (tag_idxs != 0).unsqueeze(-1).float() # [batch_size, max_tags, 1]
            sum_embs = (tag_embs * mask).sum(dim=1) # [batch_size, tag_dim]
            count_embs = mask.sum(dim=1).clamp(min=1e-6) # [batch_size, 1]
            pooled_tags = sum_embs / count_embs
            cat_features.append(pooled_tags)

        # Nối tất cả các nhánh lại
        all_features = [sbert_projected, numerical_projected] + cat_features
        concatenated_features = torch.cat(all_features, dim=1)
        
        # Đưa qua các lớp Fusion
        x = self.fusion_relu(self.fusion_dense_1(concatenated_features))
        final_embedding = self.fusion_dense_2(x)
        return final_embedding

class UserTower(nn.Module):
    """
    Tháp Người dùng (User Tower)
    Phiên bản đơn giản chỉ học embedding từ User ID.
    """
    def __init__(self, num_users, embedding_dimension):
        super().__init__()
        self.user_embedding_layer = nn.Embedding(num_users, embedding_dimension)
        
    def forward(self, user_indices):
        # Input: user_indices (Tensor các index của user)
        # Output: User Embeddings
        return self.user_embedding_layer(user_indices)

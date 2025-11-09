from torch.utils.data import Dataset
import torch
import numpy as np

# --- 3. ĐỊNH NGHĨA DATASET ---
class UserItemInteractionDataset(Dataset):
    """
    Lớp Dataset cho các tương tác User-Item.
    --- THAY ĐỔI ---
    Cần thêm user_history_map và max_len để xử lý lịch sử.
    """
    def __init__(self, interactions_df, user_history_map, max_history_length):
        self.user_indices = interactions_df['user_idx'].values
        self.item_indices = interactions_df['item_idx'].values
        self.user_history_map = user_history_map
        self.max_len = max_history_length

    def __len__(self):
        return len(self.user_indices)

    def __getitem__(self, idx):
        # Lấy user và item dương (positive item)
        user_idx = self.user_indices[idx]
        pos_item_idx = self.item_indices[idx]

        # Lấy lịch sử của user từ map
        # .get(user_idx, []) trả về list rỗng nếu user không có trong map
        history = self.user_history_map.get(user_idx, [])
        
        # --- XỬ LÝ LỊCH SỬ ---
        # 1. Loại bỏ chính item dương ra khỏi lịch sử
        # (Để model không "ăn gian" học cách copy item từ lịch sử)
        # Chuyển sang set để xóa hiệu quả (nếu lịch sử dài)
        history_set = set(history)
        history_set.discard(pos_item_idx)
        history = list(history_set)
        
        # 2. Cắt (Truncate) lịch sử nếu quá dài
        if len(history) > self.max_len:
            # Lấy self.max_len item cuối cùng (giả định là gần nhất)
            history = history[-self.max_len:]
        
        # 3. Đệm (Pad) lịch sử nếu quá ngắn
        padding_length = self.max_len - len(history)
        if padding_length > 0:
            # Pad 0 vào ĐẦU (pre-padding)
            # [0, 0, 0, item1, item2]
            history = [0] * padding_length + history
            
        # Chuyển lịch sử sang tensor
        history_tensor = torch.tensor(history, dtype=torch.long)
        
        # Trả về 3 giá trị
        return user_idx, pos_item_idx, history_tensor

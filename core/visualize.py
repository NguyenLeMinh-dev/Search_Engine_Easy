import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CẤU HÌNH ---
DATA_DIR = './data/interaction_splits/'
TRAIN_FILE = os.path.join(DATA_DIR, 'train_interactions.csv')
VAL_FILE = os.path.join(DATA_DIR, 'val_interactions.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test_interactions.csv')

# Thư mục để lưu biểu đồ
OUTPUT_DIR = './data_visualizations'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. TẢI VÀ GỘP DỮ LIỆU ---
print("Đang tải dữ liệu...")
try:
    df_train = pd.read_csv(TRAIN_FILE)
    df_val = pd.read_csv(VAL_FILE)
    df_test = pd.read_csv(TEST_FILE)
    
    # Gộp tất cả lại để phân tích tổng thể
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    
except FileNotFoundError as e:
    print(f"LỖI: Không tìm thấy file: {e.filename}")
    print("Hãy đảm bảo bạn đã chạy script `split_dataset.py` và các file tồn tại.")
    exit()
except Exception as e:
    print(f"Lỗi khi đọc file: {e}")
    exit()

print(f"Đã tải thành công {len(df_all):,} tương tác.")

# --- 2. THỐNG KÊ CƠ BẢN ---
n_users = df_all['user_id'].nunique()
n_items = df_all['item_id'].nunique()
total_interactions = len(df_all)
possible_interactions = n_users * n_items
sparsity = 1.0 - (total_interactions / possible_interactions)

print("\n--- THỐNG KÊ TỔNG QUAN ---")
print(f"  Số lượng User duy nhất:    {n_users:,}")
print(f"  Số lượng Item duy nhất:    {n_items:,}")
print(f"  Tổng số tương tác:        {total_interactions:,}")
print(f"  Độ thưa (Sparsity):        {sparsity:.6f} (Càng gần 1.0 càng thưa)")
print("---------------------------\n")


# --- 3. PHÂN TÍCH USER (Số tương tác / user) ---
print("Đang phân tích User...")
user_interaction_counts = df_all.groupby('user_id')['item_id'].count()

print(f"  Thống kê tương tác mỗi User:")
print(user_interaction_counts.describe())

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))
sns.histplot(user_interaction_counts, bins=50, kde=False)
plt.title(f'Phân bố số lượng tương tác của mỗi User (Tổng: {n_users} Users)')
plt.xlabel('Số lượng tương tác')
plt.ylabel('Số lượng Users')
plt.yscale('log') # Dùng thang log vì phân bố này thường bị lệch (long-tail)
plt.grid(axis='y', linestyle='--', alpha=0.7)
output_path = os.path.join(OUTPUT_DIR, 'user_interaction_distribution.png')
plt.savefig(output_path)
print(f"  -> Đã lưu biểu đồ User vào: {output_path}")


# --- 4. PHÂN TÍCH ITEM (Độ phổ biến / item) ---
print("\nĐang phân tích Item...")
item_interaction_counts = df_all.groupby('item_id')['user_id'].count()

print(f"  Thống kê độ phổ biến của Item:")
print(item_interaction_counts.describe())

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))
sns.histplot(item_interaction_counts, bins=50, kde=False)
plt.title(f'Phân bố độ phổ biến của Item (Tổng: {n_items} Items)')
plt.xlabel('Số lượng tương tác nhận được (Độ phổ biến)')
plt.ylabel('Số lượng Items')
plt.yscale('log') # Dùng thang log
plt.grid(axis='y', linestyle='--', alpha=0.7)
output_path = os.path.join(OUTPUT_DIR, 'item_popularity_distribution.png')
plt.savefig(output_path)
print(f"  -> Đã lưu biểu đồ Item vào: {output_path}")
print("\nHoàn tất phân tích.")
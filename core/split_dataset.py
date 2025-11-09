import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os

# --- 1. CẤU HÌNH ---
INPUT_FILE = "/home/minh/Documents/SEG_project/core/data/persona_user_interactions.csv"
OUTPUT_DIR = "/home/minh/Documents/SEG_project/core/data/interaction_splits" # Thư mục lưu kết quả

# Tỷ lệ chia (phần còn lại sẽ là test set)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1 # Tỷ lệ validation lấy từ phần còn lại sau train
# TEST_RATIO sẽ là 1.0 - TRAIN_RATIO - VAL_RATIO (trong trường hợp này là 0.1)

# Cột thời gian để sắp xếp và chia
TIMESTAMP_COL = "timestamp"

# --- 2. HÀM CHIA DỮ LIỆU ---
def time_based_split(df, timestamp_col, train_ratio, val_ratio):
    """Chia DataFrame thành train/val/test dựa trên thời gian."""
    print(f"Đang chia dữ liệu dựa trên cột '{timestamp_col}'...")

    # Đảm bảo cột timestamp là kiểu datetime
    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    except Exception as e:
        print(f"Lỗi khi chuyển đổi cột '{timestamp_col}' sang datetime: {e}")
        print("Vui lòng đảm bảo cột timestamp có định dạng đúng (YYYY-MM-DD HH:MM:SS).")
        return None, None, None

    # Sắp xếp dữ liệu theo thời gian
    df_sorted = df.sort_values(by=timestamp_col).reset_index(drop=True)
    n_total = len(df_sorted)
    print(f"Tổng số tương tác: {n_total}")

    # Tính toán các điểm chia
    train_end_idx = int(n_total * train_ratio)
    # Tính index cho validation từ phần còn lại
    remaining_count = n_total - train_end_idx
    val_end_idx = train_end_idx + int(remaining_count * (val_ratio / (1.0 - train_ratio))) # Điều chỉnh tỷ lệ val

    # Thực hiện chia
    df_train = df_sorted.iloc[:train_end_idx]
    df_val = df_sorted.iloc[train_end_idx:val_end_idx]
    df_test = df_sorted.iloc[val_end_idx:]

    print(f"  -> Train set: {len(df_train)} ({len(df_train)/n_total:.1%})")
    print(f"  -> Validation set: {len(df_val)} ({len(df_val)/n_total:.1%})")
    print(f"  -> Test set: {len(df_test)} ({len(df_test)/n_total:.1%})")

    # Kiểm tra xem có bị mất dòng nào không
    assert len(df_train) + len(df_val) + len(df_test) == n_total

    return df_train, df_val, df_test

# --- 3. HÀM CHÍNH (MAIN) ---
def main(args):
    """Hàm điều phối chính."""
    print(f"Đang đọc dữ liệu tương tác từ: {args.input_file}")
    try:
        interactions_df = pd.read_csv(args.input_file)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp '{args.input_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"Lỗi khi đọc file CSV: {e}")
        sys.exit(1)

    if TIMESTAMP_COL not in interactions_df.columns:
        print(f"Lỗi: Không tìm thấy cột timestamp '{TIMESTAMP_COL}' trong file.")
        sys.exit(1)

    # Chia dữ liệu
    df_train, df_val, df_test = time_based_split(
        interactions_df,
        TIMESTAMP_COL,
        TRAIN_RATIO,
        VAL_RATIO
    )

    if df_train is None: # Nếu có lỗi khi chia
        sys.exit(1)

    # Tạo thư mục output nếu chưa có
    os.makedirs(args.output_dir, exist_ok=True)

    # Lưu kết quả
    train_path = os.path.join(args.output_dir, "train_interactions.csv")
    val_path = os.path.join(args.output_dir, "val_interactions.csv")
    test_path = os.path.join(args.output_dir, "test_interactions.csv")

    print(f"\nĐang lưu các tập dữ liệu vào thư mục: {args.output_dir}/")
    try:
        df_train.to_csv(train_path, index=False, encoding='utf-8')
        df_val.to_csv(val_path, index=False, encoding='utf-8')
        df_test.to_csv(test_path, index=False, encoding='utf-8')
        print(f"  -> Đã lưu Train set: {train_path}")
        print(f"  -> Đã lưu Validation set: {val_path}")
        print(f"  -> Đã lưu Test set: {test_path}")
        print("\n--- HOÀN TẤT CHIA DỮ LIỆU ---")
    except Exception as e:
        print(f"Lỗi khi lưu file CSV: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chia dữ liệu tương tác thành train/val/test theo thời gian.")
    parser.add_argument(
        '--input_file',
        type=str,
        default=INPUT_FILE,
        help=f"Đường dẫn đến file tương tác đầu vào (mặc định: {INPUT_FILE})."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=OUTPUT_DIR,
        help=f"Thư mục để lưu các file train/val/test đã chia (mặc định: {OUTPUT_DIR})."
    )
    args = parser.parse_args()
    main(args)

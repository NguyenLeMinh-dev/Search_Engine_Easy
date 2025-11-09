import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_dataframe(df):
    """In ra các phân tích và thống kê cơ bản về DataFrame."""
    print("\n" + "="*50)
    print("📊 BẮT ĐẦU PHÂN TÍCH DỮ LIỆU SAU KHI LÀM SẠCH 📊")
    print("="*50)

    print("\n1️⃣ Thông tin tổng quan (Info):")
    df.info()

    print("\n" + "-"*50)
    print("\n2️⃣ Thống kê mô tả cho các cột số (Describe):")
    numeric_cols = ['rating', 'price_min', 'price_max', 'open_hour', 'close_hour', 'gps_lat', 'gps_long']
    print(df[numeric_cols].describe())

    print("\n" + "-"*50)
    print("\n3️⃣ Phân phối các quán ăn theo Quận:")
    print(df['district'].value_counts())

    print("\n" + "-"*50)
    print("\n4️⃣ Kiểm tra các giá trị rỗng (Null Values):")
    df_for_analysis = df.replace('', pd.NA)
    print(df_for_analysis.isna().sum())

    print("\n" + "="*50)
    print("📈 ĐANG VẼ BIỂU ĐỒ TRỰC QUAN HÓA DỮ LIỆU...")
    print("="*50)
    
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        # Thiết lập font hỗ trợ tiếng Việt
        plt.rcParams['font.family'] = 'DejaVu Sans'
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Phân Phối Dữ Liệu Foody Cần Thơ', fontsize=20)

        # Biểu đồ phân phối Rating
        sns.histplot(df['rating'].dropna(), kde=True, ax=axes[0, 0], color='skyblue')
        axes[0, 0].set_title('Phân Phối Điểm Rating')

        # Biểu đồ phân phối Giá tối thiểu
        sns.histplot(df[df['price_min'] < 200000]['price_min'].dropna(), kde=True, ax=axes[0, 1], color='salmon')
        axes[0, 1].set_title('Phân Phối Giá Tối Thiểu (dưới 200k)')

        # Biểu đồ số lượng quán theo Quận
        sns.countplot(y=df['district'], ax=axes[1, 0], order = df['district'].value_counts().index, palette='viridis', hue=df['district'], legend=False)
        axes[1, 0].set_title('Số Lượng Quán Ăn Theo Quận')
        
        axes[1, 1].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        print("✅ Biểu đồ đã được tạo. Vui lòng xem cửa sổ mới hiển thị.")
        plt.show()

    except ImportError:
         print("\nVui lòng cài đặt các thư viện cần thiết: pip install matplotlib seaborn")
    except Exception as e:
        print(f"Lỗi khi vẽ biểu đồ: {e}")


if __name__ == "__main__":
    # File CSV đã được làm sạch từ script trước
    cleaned_csv_path = r"final_processed_data.csv"

    if not os.path.exists(cleaned_csv_path):
        print(f"Lỗi: Không tìm thấy file '{cleaned_csv_path}'.")
        print("Vui lòng chạy file 'clean_foody_script.py' trước để tạo ra file này.")
    else:
        print(f"Đọc dữ liệu từ file '{cleaned_csv_path}'...")
        df_cleaned = pd.read_csv(cleaned_csv_path)
        analyze_dataframe(df_cleaned)

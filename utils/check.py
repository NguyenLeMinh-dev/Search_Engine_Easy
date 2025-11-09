import pandas as pd
import os
# ==== CONFIG ====
PATH = os.path.dirname(os.path.abspath(__file__))
file_path = "/home/minh/Documents/SEG_project/datas/label/labeled_search_results_quán_cơm_văn_phòng.csv"
query = "món_ngon_quận_1"

# ==== ĐỌC FILE ====
df = pd.read_csv(file_path)

# ==== HIỂN THỊ TỔNG QUAN ====
print(f"🔍 Kiểm tra nhãn cho query: '{query}'")
print(f"📂 Tổng số dòng: {len(df)}\n")

# ==== THỐNG KÊ PHÂN BỐ ====
print("📊 Phân bố nhãn (tỷ lệ %):")
print(df["llm_label"].value_counts(normalize=True).sort_index().map(lambda x: f"{x:.1%}"))
print("-" * 80)

# ==== CHỌN CHẾ ĐỘ HIỂN THỊ ====
mode = input("Chọn chế độ (1 = ngẫu nhiên, 2 = nhãn thấp, 3 = nhãn cao, 4 = nhãn -1 lỗi): ")

if mode == "1":
    sample_df = df.sample(10, random_state=42)
elif mode == "2":
    sample_df = df[df["llm_label"] <= 1].head(72)
elif mode == "3":
    sample_df = df[df["llm_label"] >= 2].head(52)
elif mode == "4":
    sample_df = df[df["llm_label"] == -1].head(50)
else:
    sample_df = df.sample(10)

# ==== DUYỆT VÀ SỬA NHÃN ====
for i, row in sample_df.iterrows():
    print(f"\n📍 ID: {row['id']}")
    print(f"🍽️ Tên quán: {row['name']}")
    print(f"💬 Bình luận: {row['comments']}")
    print(f"🧠 Nội dung: {row['text_for_embedding'][:150]}...")
    print(f"🏷️ Nhãn hiện tại: {row['llm_label']}")

    # Nhập nhãn mới (0,1,2,3) hoặc Enter để giữ nguyên
    new_label = input("Nhập nhãn mới (0/1/2/3) hoặc Enter để giữ nguyên: ")
    if new_label in ["0", "1", "2", "3"]:
        df.at[i, "llm_label"] = int(new_label)
        print(f"✅ Nhãn đã được cập nhật thành {new_label}")
    else:
        print("ℹ️ Giữ nguyên nhãn cũ")

    print("-" * 100)

# ==== LƯU FILE ====
save_choice = input("Bạn có muốn lưu lại file CSV không? (y/n): ")
if save_choice.lower() == "y":
    df.to_csv(file_path, index=False)
    print(f"💾 File đã được lưu tại {file_path}")
else:
    print("❌ File chưa được lưu, các thay đổi chỉ tồn tại trong phiên làm việc này.")

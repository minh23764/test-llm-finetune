from pathlib import Path
from datasets import load_dataset

# 1. Xác định đường dẫn tới file train.jsonl
BASE_DIR = Path(__file__).parent.parent   # thư mục gốc project
DATA_PATH = BASE_DIR / "data" / "train.jsonl"

def main():
    print("Đọc dataset từ:", DATA_PATH)

    # 2. Load dataset từ file JSONL
    dataset = load_dataset("json", data_files=str(DATA_PATH))
    train_ds = dataset["train"]

    # 3. In thử vài mẫu
    print("Số mẫu:", len(train_ds))
    print("Mẫu đầu tiên:")
    print(train_ds[0])

if __name__ == "__main__":
    main()

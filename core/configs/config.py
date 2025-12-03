import os
import torch


PROCESSED_ITEM_DATA_DIR = './core/data/processed_item_data'

INTERACTION_SPLITS_DIR = './core/data/interaction_splits'
TRAIN_INTERACTIONS_FILE = os.path.join(INTERACTION_SPLITS_DIR, 'train_interactions.csv')
VAL_INTERACTIONS_FILE = os.path.join(INTERACTION_SPLITS_DIR, 'val_interactions.csv')

# Hyperparameters
EMBEDDING_DIMENSION = 128
BATCH_SIZE = 1024
NUM_EPOCHS = 300
# LEARNING_RATE = 0.001 # Giảm LR để giải quyết vấn đề loss bị "kẹt"
LEARNING_RATE = 1e-4 
NUM_NEG_SAMPLES = 4
# K cho đánh giá Recall@K trên tập validation
EVAL_K = 100


DROPOUT_RATE = 0.5
MAX_HISTORY_LENGTH = 50 # Độ dài tối đa của lịch sử user (dùng để padding)
TEMPERATURE = 0.1 # Thêm nhiệt độ cho InfoNCE Loss
# ------------------------------


BEST_MODEL_DIR = 'models_v3_finetune'


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

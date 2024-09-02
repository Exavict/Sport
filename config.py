import torch


DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# LSTM parameters
INPUT_DIM = 34  # 17 keypoints (x, y)
HIDDEN_DIM = 8  # hidden layers
NUM_LAYERS = 2
OUTPUT_DIM = 3  # sit-up; pushup; squat
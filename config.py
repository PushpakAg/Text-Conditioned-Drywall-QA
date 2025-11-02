import torch
DATA_DIR = "data/processed"
TEXT_MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16 
NUM_WORKERS = 4  

FREEZE_ENCODERS = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

CHECKPOINT_DIR = "outputs/checkpoints"
LOGS_DIR = "outputs/logs"
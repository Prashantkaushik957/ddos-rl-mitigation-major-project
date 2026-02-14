"""
Central configuration for the DDoS RL Mitigation project.
All hyperparameters, paths, and constants are defined here.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
PAPER_FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"

# Create directories if they don't exist
for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, LOGS_DIR, PAPER_FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Dataset Configuration
# ──────────────────────────────────────────────
DATASET_NAME = "CICDDoS2019"
LABEL_COLUMN = "Label"
ATTACK_TYPES = [
    "BENIGN",
    "DDoS",
    "DrDoS_DNS",
    "DrDoS_LDAP",
    "DrDoS_MSSQL",
    "DrDoS_NTP",
    "DrDoS_NetBIOS",
    "DrDoS_SNMP",
    "DrDoS_SSDP",
    "DrDoS_UDP",
    "Syn",
    "TFTP",
    "UDPLag",
    "WebDDoS",
    "Portmap",
]

# Binary label mapping: 0 = Benign, 1 = Attack
BINARY_LABELS = {"BENIGN": 0}  # All others default to 1

# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────
NUM_SELECTED_FEATURES = 20
AUTOENCODER_LATENT_DIM = 10
NORMALIZATION_METHOD = "minmax"  # "minmax" or "standard"
TEST_SPLIT_RATIO = 0.2
VAL_SPLIT_RATIO = 0.1
RANDOM_SEED = 42
MAX_SAMPLES_PER_CLASS = 50000  # For undersampling majority class

# ──────────────────────────────────────────────
# Autoencoder (Feature Extraction)
# ──────────────────────────────────────────────
AE_HIDDEN_DIMS = [64, 32]
AE_LEARNING_RATE = 1e-3
AE_BATCH_SIZE = 256
AE_EPOCHS = 50
AE_PATIENCE = 10  # Early stopping patience

# ──────────────────────────────────────────────
# SDN Environment
# ──────────────────────────────────────────────
NUM_ACTIONS = 3  # ALLOW=0, RATE_LIMIT=1, DROP=2
ACTION_NAMES = {0: "ALLOW", 1: "RATE_LIMIT", 2: "DROP"}
BATCH_SIZE_ENV = 512  # Flows per episode
FALSE_POSITIVE_THRESHOLD = 0.15  # Episode terminates if FP rate exceeds this

# Reward structure
REWARD_CORRECT_BENIGN = 1.0      # Correctly ALLOW benign traffic
REWARD_CORRECT_ATTACK = 2.0      # Correctly DROP/RATE_LIMIT attack
REWARD_FALSE_POSITIVE = -3.0     # Incorrectly DROP/RATE_LIMIT benign
REWARD_FALSE_NEGATIVE = -1.0     # Incorrectly ALLOW attack
REWARD_RATE_LIMIT_ATTACK = 1.5   # Partial credit for rate-limiting (not dropping) attack

# ──────────────────────────────────────────────
# DQN / DDQN Hyperparameters
# ──────────────────────────────────────────────
DQN_LEARNING_RATE = 1e-4
DQN_GAMMA = 0.99
DQN_EPSILON_START = 1.0
DQN_EPSILON_END = 0.01
DQN_EPSILON_DECAY_STEPS = 50000
DQN_REPLAY_BUFFER_SIZE = 100000
DQN_BATCH_SIZE = 64
DQN_TARGET_UPDATE_TAU = 0.005
DQN_HIDDEN_DIMS = [128, 64]

# ──────────────────────────────────────────────
# PPO Hyperparameters
# ──────────────────────────────────────────────
PPO_LEARNING_RATE = 3e-4
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_RATIO = 0.2
PPO_ENTROPY_COEF = 0.01
PPO_VALUE_LOSS_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5
PPO_UPDATE_EPOCHS = 10
PPO_BATCH_SIZE = 64
PPO_ROLLOUT_LENGTH = 2048
PPO_HIDDEN_DIMS = [128, 64]

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
NUM_EPISODES = 500
CHECKPOINT_INTERVAL = 10  # Save model every N episodes
LOG_INTERVAL = 5          # Print stats every N episodes
EARLY_STOPPING_PATIENCE = 30  # Stop if no improvement for N episodes

# ──────────────────────────────────────────────
# Baselines (Traditional ML)
# ──────────────────────────────────────────────
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None
SVM_KERNEL = "rbf"
SVM_C = 1.0
XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1

# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
EVALUATION_METRICS = ["accuracy", "precision", "recall", "f1", "auc_roc"]
CONFUSION_MATRIX_FIGSIZE = (10, 8)
COMPARISON_FIGSIZE = (12, 6)

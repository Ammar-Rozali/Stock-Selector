import os
import time
from tensorflow.keras.layers import LSTM

# Window size or the sequence length
N_STEPS = 90
# Lookup step, 1 is the next day
LOOKUP_STEP = 15

# test ratio size, 0.2 is 20%
TEST_SIZE = 0.2
# features to use
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]

# date now
date_now = time.strftime("%Y-%m-%d")

# model parameters

N_LAYERS = 3
# LSTM cell
CELL = LSTM
# 256 LSTM neurons
UNITS = 256
# 40% dropout
DROPOUT = 0.4
# whether to use bidirectional RNNs
BIDIRECTIONAL = False

# training parameters

# mean absolute error loss
# LOSS = "mae"
# huber loss
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 800

# stock market
ticker = "7160.kl"

# create csv file
ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")

model_name = f"{date_now}_{ticker}-Loss-{LOSS}-step-{LOOKUP_STEP}-epoch{EPOCHS}"


if BIDIRECTIONAL:
    model_name += "-b"

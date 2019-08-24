import os

#File Paths
PROJECT_DIR="/home/name/Eng2C++/"
TXT_DATA=os.path.join(PROJECT_DIR,"data/test_data.txt")
ENCODER_PATH=os.path.join(PROJECT_DIR,"model/encoder.pth")
DECODER_PATH=os.path.join(PROJECT_DIR,"model/decoder.pth")

#Special Indices
SOS_TOKEN = 0
EOS_TOKEN = 1

#Languages
SOURCE = "c++"
TARGET = "python"
MAX_LENGTH = 20

#Hyperparameters
TEACHER_FORCE_RATIO = 0.5
LEARNING_RATE = 0.01
N_INTERATIONS = 200
HIDDEN_SIZE = 256
EVAL_PERCENTAGE = 0.3

#Cosmetics
AVG_EVERY = 50

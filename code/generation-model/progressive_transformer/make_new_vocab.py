from vocabulary import *

import os 
train_path = "Data/augmented_data/enhanced-gloss/train/"
for fname in os.listdir(train_path):
    input_file = open(train_path + fname, "r").readlines()
    vocab_ = Vocabulary()

    for line in input_file:
        vocab_._from_list(line.split())
    vocab_.to_file("Configs/src_%s_vocab.txt"%fname)
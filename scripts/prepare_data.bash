#!/bin/bash

echo "Preparing training data..."
python3 src/data/data_processing.py \
    --data_path ./Data/LA/ASVspoof2019_LA_train/flac \
    --output_path ./Data/train.pkl \
    --label_path ./Data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt

echo "Preparing dev data..."
python3 src/data/data_processing.py \
    --data_path ./Data/LA/ASVspoof2019_LA_dev/flac \
    --output_path ./Data/dev.pkl \
    --label_path ./Data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt 

echo "Preparing test data..."
python3 src/data/data_processing.py \
    --data_path ./Data/LA/ASVspoof2019_LA_eval/flac \
    --output_path ./Data/eval.pkl \
    --label_path ./Data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt
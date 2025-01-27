import os
import random
import pickle
import argparse
import soundfile as sf
from tqdm import tqdm

# Assuming CQCC and MFCC are imported from the refactored src.features
from src.features.cqcc import cqcc
from src.features.extract import extract_mfcc

def process_data(data_path, label_path, output_path, max_files=None):
    # Read in labels
    filename2label = {}
    with open(label_path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            filename, label = tokens[1], tokens[-1]
            filename2label[filename] = label

    all_files = [f for f in os.listdir(data_path) if f.endswith('.flac')]
    random.shuffle(all_files)
    
    if max_files:
        all_files = all_files[:max_files]

    feats = []
    
    for filename in tqdm(all_files, desc="Processing Audio Files"):
        file_id = filename.split('.')[0]
        if file_id not in filename2label:
            continue  # Skip files not in the protocol
            
        label = filename2label[file_id]
        filepath = os.path.join(data_path, filename)
        
        sig, rate = sf.read(filepath)
        
        # Note: CQCC parameters are hardcoded here based on your original spec
        fmax = rate / 2
        fmin = fmax / 2**9
        feat_cqcc, _, _, _, _, _, _ = cqcc(sig, rate, 96, fmax, fmin, 16, 19, 'ZsdD')
        
        # MFCC Extraction using the refactored pipeline
        feat_mfcc = extract_mfcc(sig, sr=rate)
        
        feats.append((feat_cqcc.T, feat_mfcc, label))

    print(f"Saved {len(feats)} instances to {output_path}")
    with open(output_path, 'wb') as outfile:
        pickle.dump(feats, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--label_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    args = parser.parse_args()
    
    process_data(args.data_path, args.label_path, args.output_path)
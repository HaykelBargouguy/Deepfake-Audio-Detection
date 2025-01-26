import pandas as pd
import soundfile as sf
import numpy as np
import torch
from transformers import AutoFeatureExtractor, WavLMForXVector, UniSpeechSatForXVector
from tqdm import tqdm

def envelope(y, rate, threshold=0.0005):
    """Voice Activity Detection (VAD) based on rolling mean amplitude."""
    mask = []
    y_series = pd.Series(y).apply(np.abs)
    y_mean = y_series.rolling(window=int(rate / 10), min_periods=1, center=True).mean()
    for mean in y_mean:
        mask.append(mean > threshold)
    return mask

def clean_audio(file_path):
    """Loads audio and applies VAD mask."""
    signal, rate = sf.read(file_path)
    mask = envelope(signal, rate)
    signal = signal[mask]
    return signal, rate

def extract_transformer_embeddings(samples_dict, model_type="wavlm"):
    """
    Extracts deep embeddings using either WavLM or UniSpeech-SAT.
    Args:
        samples_dict: Dict mapping label/key to list of filepaths.
        model_type: 'wavlm' or 'unispeech'
    """
    feats = {key: [] for key in samples_dict}
    total_samples = sum(len(paths) for paths in samples_dict.values())
    
    if model_type == "wavlm":
        model_name = "microsoft/wavlm-base-plus-sv"
        model_class = WavLMForXVector
    elif model_type == "unispeech":
        model_name = "microsoft/unispeech-sat-base-plus-sv"
        model_class = UniSpeechSatForXVector
    else:
        raise ValueError("Invalid model_type. Choose 'wavlm' or 'unispeech'.")

    print(f"Loading {model_name}...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name)
    model.eval()

    with tqdm(total=total_samples, desc=f"Extracting {model_type} embeddings") as pbar:
        for key, filepaths in samples_dict.items():
            for filepath in filepaths:
                signal, rate = clean_audio(filepath)
                inputs = feature_extractor(signal, sampling_rate=rate, return_tensors="pt")
                
                with torch.no_grad():
                    embeddings = model(**inputs).embeddings
                
                # Normalize and convert to numpy
                embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu().numpy()
                feats[key].append(embeddings)
                pbar.update(1)
            
            feats[key] = np.array(feats[key])

    return feats
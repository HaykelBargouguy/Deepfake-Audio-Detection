from dataclasses import dataclass
import torch

@dataclass
class ProjectConfig:
    # General Data Parameters
    logical_data_root: str = '../Data/LA/'
    physical_data_root: str = '../Data/PA/'
    sample_rate: int = 16000
    max_len: int = 64000
    
    # Deep Learning Training Parameters
    num_epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.0001
    lr_decay: float = 0.5
    weight_decay: float = 0.0005
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model & Loss Parameters
    feature_type: str = 'lfcc' # Options: 'mfcc', 'lfcc', 'cqcc', 'spect'
    enc_dim: int = 256
    add_loss: str = "ocsoftmax" # Options: 'softmax', 'amsoftmax', 'ocsoftmax'
    r_real: float = 0.9
    r_fake: float = 0.2
    alpha: float = 20.0
    
    # ML Baseline Parameters
    ml_max_len: int = 50
    
    # Output Paths
    out_fold: str = './outputs/models/'
import torch
import torch.nn as nn
import math

from Models import MFCCTransformer

# Insérez ici la définition de PositionalEncoding et MFCCTransformer

# Initialisation du modèle
model = MFCCTransformer(input_dim=1024, d_model=512, nhead=8, num_encoder_layers=3, dim_feedforward=2048, num_classes=2)

# Création d'un tensor d'entrée aléatoire
input_tensor = torch.randn(32, 60, 1024)  # Batch size = 32, Seq length = 60, Feature size = 1024

# Passage du tensor à travers le modèle
output = model(input_tensor)

# Affichage de la sortie
print(output.shape)  # Devrait afficher torch.Size([32, 10])

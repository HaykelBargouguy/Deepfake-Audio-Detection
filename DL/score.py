from torch import Tensor
from dataset import SoundDataset, pad
from torch.utils.data import DataLoader
from embedding import *
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
transforms = transforms.Compose([
    lambda x: pad(x),
    lambda x: librosa.util.normalize(x),
    lambda x: Tensor(x)
])

def test_model_ocsoftmax(feat_model_path, loss_model_path,part, add_loss, device, transforms = transforms):
    dirname = os.path.dirname
    basename = os.path.splitext(os.path.basename(feat_model_path))[0]
    model_feats = []
    model_labels = []
    count = 0
    prev_label = None
    if "checkpoint" in dirname(feat_model_path):
        dir_path = dirname(dirname(feat_model_path))
    else:
        dir_path = dirname(feat_model_path)
    model = torch.load(feat_model_path, map_location='cuda').to(device)
    loss_model = torch.load(loss_model_path, map_location='cuda').to(device)
    is_eval = (part == 'eval')

 
    test_dataset =  SoundDataset(data_root = "../Data/PA/" , transform = transforms , feature_type='lfcc' ,is_logical=False, is_train=False, is_eval=True , random_sample = 40000 )

    testDataLoader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model.eval()

    with open(os.path.join('resultats/', 'lfcc_resnet_18.txt'), 'w') as cm_score_file:
        for i, (lfcc, labels) in enumerate(tqdm(testDataLoader)):
            lfcc = lfcc.unsqueeze(1).float().to(device)
            labels = labels.to(device)
            feats, lfcc_outputs = model(lfcc)

            score = F.softmax(lfcc_outputs)[:, 0]

            if add_loss == "ocsoftmax":
                ang_isoloss, score = loss_model(feats, labels)
            elif add_loss == "amsoftmax":
                outputs, moutputs = loss_model(feats, labels)
                score = F.softmax(outputs, dim=1)[:, 0]

            for j in range(labels.size(0)):
                if count <= 300:
                    if prev_label != labels[j]:
                        model_feats.append(feats[j].detach().cpu().numpy())
                        model_labels.append("spoof" if labels[j].data.cpu().numpy() else "bonafide")
                        prev_label = labels[j]
                        count = count + 1
                cm_score_file.write(
                    '%s %s \n' % ( "spoof" if labels[j].data.cpu().numpy() else "bonafide", score[j].item()))

    output_file = os.path.join('resultats/', 'feats_' + 'lfcc_resnet-18' + ".tsv")
    meta_file = os.path.join('resultats/', 'meta_' + 'lfcc_resnet_18' + ".tsv")

    with open(output_file, 'w') as f:
        for feat in model_feats:
            sample_str = '\t'.join([str(e) for e in feat])
            f.write(f"{sample_str}\n")

    with open(meta_file, 'w') as meta:
        for label in model_labels:
            meta.write(f"{label}\n")


def test_ocsoftmax(model_dir, model, loss_model, add_loss, device):
    model_path = os.path.join(model_dir, model)
    loss_model_path = os.path.join(model_dir, loss_model)
    test_model_ocsoftmax(model_path, loss_model_path, "eval", add_loss, device)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = 'models/resnet/'
model = 'anti-spoofing_lfcc_model.pt'
loss_model = 'anti-spoofing_loss_model.pt'
add_loss = 'ocsoftmax'
test_ocsoftmax(model_dir, model, loss_model, add_loss, device )
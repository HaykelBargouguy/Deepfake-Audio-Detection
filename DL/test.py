import os

model_tag='model'
model_checkpoint_path = os.path.join('models', model_tag, 'model_snapshots/')
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
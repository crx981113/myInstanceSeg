import clip
from datasets.text_label import obj_text_label
import torch


device='cuda'
clip_model = 'ViT-B/32'
clip_model, preprocess = clip.load(clip_model, device=device)
with torch.no_grad():
    obj_text_inputs = torch.cat([clip.tokenize(obj_text[1]) for obj_text in obj_text_label])
    obj_text_embedding = clip_model.encode_text(obj_text_inputs.to(device)).float()
    print('stop')
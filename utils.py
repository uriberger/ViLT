from vilt.config import config as vilt_config
from vilt.modules import ViLTransformerSS
import torch

def load_model(checkpoint_path):
    config = vilt_config()
    config['load_path'] = checkpoint_path
    model = ViLTransformerSS(config)
    return model

def get_text_features(model, tokenizer, sentence):
    tokenized_input = tokenizer(sentence, return_tensors = 'pt')
    batch = {}
    batch['text_ids'] = tokenized_input['input_ids']
    batch['text_masks'] = tokenized_input['attention_mask']
    batch['text_labels'] = ''

    # Blank image
    image = torch.ones(1, 1, 3, 224, 224)
    batch['image'] = image

    res = model.infer(batch)
    return res['text_feats']

from vilt.config import config as vilt_config
from vilt.modules import ViLTransformerSS
from transformers import AutoTokenizer
import torch

def load_model(model_path):
    config = vilt_config()
    config['load_path'] = model_path
    config['tokenizer'] = 'bert-base-uncased'
    config['vocab_size'] = 30522
    config['max_text_len'] = 512
    model = ViLTransformerSS(config)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

def extract_embeddings(sentences, model, tokenizer):
    with torch.no_grad():
        input_ids = tokenizer(sentences, padding=True, return_tensors='pt').input_ids
    padded_embeddings = model.text_embeddings(input_ids)
    embeddings = []
    max_sent_len = padded_embeddings.shape[1]

    for i in range(len(sentences)):
        for j in range(max_sent_len-1, -1, -1):
            if input_ids[i, j] != 0: # pad id is 0
                embeddings.append(padded_embeddings[i, :j+1, :])
                break

    return embeddings

def extract_features(sentences, model, tokenizer):
    with torch.no_grad():
        tokenized_input = tokenizer(sentences, padding=True, return_tensors='pt')
        batch = {}
        batch['text_ids'] = tokenized_input.input_ids
        batch['text_masks'] = tokenized_input.attention_mask
        batch['text_labels'] = ''
        batch['image'] = torch.ones(1, len(sentences), 3, 224, 224)
        res = model.infer(batch)
    return res['text_feats']

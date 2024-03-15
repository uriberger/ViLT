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

def agg_feature_vectors(feature_vectors, method):
    if method == 'mean':
        return torch.mean(feature_vectors, dim=0)
    elif method == 'first':
        return feature_vectors[0]
    elif method == 'last':
        return feature_vectors[-1]
    else:
        assert False, f'Unknown feature aggregation method: {method}'

def extract_features(sentences, model, tokenizer, agg_subtokens_method=None):
    with torch.no_grad():
        tokenized_input = tokenizer(sentences, padding=True, return_tensors='pt')
        batch = {}
        batch['text_ids'] = tokenized_input.input_ids.to(model.device)
        batch['text_masks'] = tokenized_input.attention_mask.to(model.device)
        batch['text_labels'] = tokenized_input.input_ids.to(model.device)
        batch['image'] = torch.ones(1, len(sentences), 3, 224, 224).to(model.device)
        
        res = model.infer(batch)

    text_feats = res['text_feats']
    feature_list = []
    for sent_ind in range(len(sentences)):
        token_num = torch.sum(batch['text_masks'][sent_ind]).item()
        if agg_subtokens_method is not None:
            cur_token_start_ind = None
            feature_vectors = []
            for i, text_id in enumerate(batch['text_ids'][sent_ind]):
                if text_id == 101:
                    continue
                if text_id == 102:
                    break
                id_str = tokenizer.decode(text_id)
                if id_str.startswith('##'):
                    continue
                elif cur_token_start_ind is not None:
                    feature_vector = agg_feature_vectors(text_feats[sent_ind, cur_token_start_ind:i, :], agg_subtokens_method)
                    feature_vectors.append(feature_vector)
                cur_token_start_ind = i
            feature_vector = agg_feature_vectors(text_feats[sent_ind, cur_token_start_ind:token_num, :], agg_subtokens_method)
            feature_vectors.append(feature_vector)

            feature_vectors = [x.unsqueeze(dim=0) for x in feature_vectors]
            feature_list.append(torch.cat(feature_vectors, dim=0))
        else:
            feature_list.append(text_feats[sent_ind, :token_num, :])
    return feature_list

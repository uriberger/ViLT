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

def extract_features_from_sentences(sentences, model, tokenizer, agg_subtokens_method):
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
        cur_token_start_ind = None
        feature_vectors = []
        for i, text_id in enumerate(batch['text_ids'][sent_ind]):
            if text_id.item() == 101:
                continue
            if text_id.item() == 102:
                break
            id_str = tokenizer.decode(text_id)
            if id_str.startswith('##'):
                continue
            if id_str == "'" and i < len(batch['text_ids'][sent_ind]) - 1 and tokenizer.decode(batch['text_ids'][sent_ind][i+1]) == 's':
                continue
            if i < len(batch['text_ids'][sent_ind]) - 1 and tokenizer.decode(batch['text_ids'][sent_ind][i+1]) == '-':
                continue
            if id_str == '-':
                continue
            elif cur_token_start_ind is not None:
                feature_vector = agg_feature_vectors(text_feats[sent_ind, cur_token_start_ind:i, :], agg_subtokens_method)
                feature_vectors.append(feature_vector)
            cur_token_start_ind = i
        feature_vector = agg_feature_vectors(text_feats[sent_ind, cur_token_start_ind:i, :], agg_subtokens_method)
        feature_vectors.append(feature_vector)

        feature_vectors = [x.unsqueeze(dim=0) for x in feature_vectors]
        feature_list.append(torch.cat(feature_vectors, dim=0))
        
    return feature_list

def extract_features_from_tokens(token_lists, model, tokenizer, agg_subtokens_method):
    sentences = [' '.join(x) for x in token_lists]
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
        token_ind = 0
        cur_token = ''
        failed = False
        prev_token_end_ind = 0
        feature_vectors = []
        for i, text_id in enumerate(batch['text_ids'][sent_ind]):
            if text_id.item() == 101:
                continue
            if text_id.item() == 102:
                break
            id_str = tokenizer.decode(text_id)
            if id_str.startswith('##'):
                cur_token += id_str[2:]
            else:
                cur_token += id_str

            if cur_token.lower() == token_lists[sent_ind][token_ind].lower():
                feature_vector = agg_feature_vectors(text_feats[sent_ind, prev_token_end_ind+1:i+1, :], agg_subtokens_method)
                feature_vectors.append(feature_vector)
                prev_token_end_ind = i
                token_ind += 1
                cur_token = ''
            elif len(cur_token) > len(token_lists[sent_ind][token_ind]):
                assert cur_token.lower().startswith(token_lists[sent_ind][token_ind].lower()), f'Something wrong in the following sentence: {token_lists[sent_ind]} in token number {token_ind}, i.e. {token_lists[sent_ind][token_ind]}'
                feature_list.append(None)
                failed = True
                break

        if not failed:
            feature_vectors = [x.unsqueeze(dim=0) for x in feature_vectors]
            feature_list.append(torch.cat(feature_vectors, dim=0))

    return feature_list

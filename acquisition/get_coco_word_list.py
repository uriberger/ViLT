import json
from transformers import AutoTokenizer

def get_coco_tokenized_captions(json_path):
    with open(json_path, 'r') as fp:
        data = json.load(fp)['images']

    captions = []
    for sample in data:
        for sent in sample['sentences']:
            captions.append(sent['raw'])

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    encoded_captions = tokenizer(captions).input_ids
    tokenized_captions = []
    for encoded_caption in encoded_captions:
        tokenized_caption = []
        cur_word_ids = []
        for token_id in encoded_caption:
            cur_decoded_token = tokenizer.decode(token_id, skip_special_tokens=True)
            if len(cur_decoded_token) == 0:
                continue
            if not cur_decoded_token.startswith('##') and len(cur_word_ids) > 0:
                tokenized_caption.append(tokenizer.decode(cur_word_ids))
                cur_word_ids = []
            cur_word_ids.append(token_id)
        if len(cur_word_ids) > 0:
            tokenized_caption.append(tokenizer.decode(cur_word_ids))
        tokenized_captions.append(tokenized_caption)

    return tokenized_captions

def get_coco_word_list(json_path):
    tokenized_captions = get_coco_tokenized_captions(json_path)
    
    word_dict = {}
    for caption in tokenized_captions:
        for word in caption:
            word_dict[word] = True

    return list(word_dict.keys())
            

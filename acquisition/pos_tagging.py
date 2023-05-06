from acquisition.collect_flickr_data import collect_flickr_data
from embed import load_model, extract_features
import time
from acquisition.config import cache_dir, class_to_pos_tag
from acquisition.generate_nlp_data import generate_nlp_data
import os
import torch
import math

def get_pos_tag_to_class():
    res = {}
    for class_ind in range(len(class_to_pos_tag)):
        for pos_tag in class_to_pos_tag[class_ind]:
            res[pos_tag] = class_ind
    return res

def prepare_pos_tagging_data():
    nlp_data = generate_nlp_data()
    pos_tag_to_class = get_pos_tag_to_class()
    pos_tagging_data = [{
        'text': sample['text'],
        'start_position': sample['start_position'],
        'label': pos_tag_to_class[sample['pos']]
        } for sample in nlp_data]
    return pos_tagging_data

def generate_flickr_features():
    output_file_path = os.path.join(cache_dir, 'flickr_features')
    if os.path.isfile(output_file_path):
        return torch.load(output_file_path)
    else:
        sentences = collect_flickr_data()
        print('Loading model...', flush=True)
        model, tokenizer = load_model()

        # Batches
        batch_size = 10
        batch_num = math.ceil(len(sentences)/batch_size)

        res = []
        checkpoint_len = 10
        t = time.time()
        for batch_ind in range(batch_num):
            if batch_ind % checkpoint_len == 0:
                print('Starting batch ' + str(batch_ind) + ' out of ' + str(batch_num) + ', time from prev ' + str(time.time() - t), flush=True)
                t = time.time()
                torch.save(res, output_file_path)
            batch_start = batch_ind * batch_size
            batch_end = min((batch_ind + 1) * batch_size, len(sentences))
            batch = sentences[batch_start:batch_end]
            res.append(extract_features(batch, model, tokenizer))
        print('Finished! Saving', flush=True)
        torch.save(res, output_file_path)

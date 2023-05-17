from acquisition.collect_flickr_data import collect_flickr_data
from embed import load_model, extract_features
import time
from acquisition.config import cache_dir, class_to_pos_tag
from acquisition.generate_nlp_data import generate_nlp_data
import os
import torch
import math
from collections import defaultdict
from acquisition.classifier import create_model
from acquisition.trainer import create_trainer

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

def collect_pos_word_lists(pos_data):
    pos_to_word_list = {i: [] for i in range(len(class_to_pos_tag))}
    word_to_sample_inds = defaultdict(list)
    for i in range(len(pos_data)):
        sample = pos_data[i]
        for token in sample:
            word = token['text'].lower()
            label = token['label']
            pos_to_word_list[label].append(word)
            word_to_sample_inds[word].append(i)
    
    pos_to_word_list = {i: list(set(pos_to_word_list[i])) for i in range(len(class_to_pos_tag))}
    return pos_to_word_list, word_to_sample_inds

def prepare_training_data(pos_to_word_list, word_to_sample_inds, sample_num):
    class_to_data_splits = []
    for class_ind in range(len(class_to_pos_tag)):
        word_sample_num = [(x, len(word_to_sample_inds[x])) for x in pos_to_word_list[class_ind]]
        word_sample_num.sort(key=lambda x:x[1])
        words_by_rarity = [x[0] for x in word_sample_num]

        val_inds = []
        val_words = []
        while len(val_inds) < 0.1*sample_num:
            cur_word = words_by_rarity[0]
            val_inds += word_to_sample_inds[cur_word]
            val_inds = list(set(val_inds))
            val_words.append(cur_word)
            words_by_rarity = words_by_rarity[1:]

        val_inds_dict = {i: True for i in val_inds}
        train_inds = [i for i in range(len(sample_num)) if i not in val_inds_dict]
        class_to_data_splits.append({
            'train_inds': train_inds,
            'val_inds': val_inds,
            'val_words': val_words
            })
        
    return class_to_data_splits

def train_classifier(config, training_data):
    model = create_model(config)
    trainer = create_trainer(config, training_data)
    trainer.train()
    accuracy = trainer.evaluate()
    return accuracy

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

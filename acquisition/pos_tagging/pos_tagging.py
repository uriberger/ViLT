from acquisition.collect_flickr_data import collect_flickr_data
from embed import load_model, extract_features
from acquisition.config import cache_dir, flickr_json_path
from acquisition.pos_tagging.generate_pos_data import generate_pos_data
import os
import torch
import math
from acquisition.classifier import create_classifier
from acquisition.trainer import create_trainer
from tqdm import tqdm
import random

def generate_features(model_path, sentences):
    output_file_path = os.path.join(cache_dir, 'flickr_features')
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    if os.path.isfile(output_file_path):
        res = torch.load(output_file_path)
    else:
        res = []

    print('Loading model...', flush=True)
    model, tokenizer = load_model(model_path)
    model.to(torch.device('cuda'))

    # Batches
    batch_size = 4
    first_batch = len(res)
    batch_num = math.ceil(len(sentences)/batch_size)

    checkpoint_len = 10
    for batch_ind in tqdm(range(first_batch, batch_num)):
        if batch_ind % checkpoint_len == 0:
            torch.save(res, output_file_path)
        batch_start = batch_ind * batch_size
        batch_end = min((batch_ind + 1) * batch_size, len(sentences))
        batch = sentences[batch_start:batch_end]
        res += extract_features(batch, model, tokenizer, agg_subtokens_method='mean')
    print('Finished! Saving', flush=True)
    torch.save(res, output_file_path)

    return res

def get_data(model_path):
    sentences = collect_flickr_data(flickr_json_path, split='test')
    flickr_features = generate_features(model_path, sentences)
    flickr_pos_data = generate_pos_data(sentences)

    assert len(flickr_features) == len(flickr_pos_data)

    # Features and pos data were created using different tokenizers, filter sentences that were tokenized differently
    data = []
    for feature_vectors, pos_data in zip(flickr_features, flickr_pos_data):
        if feature_vectors.shape[0] != len(pos_data):
            continue
        data += [(feature_vectors[i], pos_data[i]['label']) for i in range(len(pos_data))]

    random.shuffle(data)
    train_sample_num = int(0.8*len(data))
    train_data = data[:train_sample_num]
    test_data = data[train_sample_num:]

    return train_data, test_data

def train_classifier(model_path, classifier_config):
    train_data, test_data = get_data(model_path)
    classifier = create_classifier(classifier_config)
    trainer = create_trainer(classifier, classifier_config, train_data, test_data)
    trainer.train()
    accuracy, res_mat = trainer.evaluate()
    return accuracy, res_mat

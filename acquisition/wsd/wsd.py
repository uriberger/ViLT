from acquisition.classifier import create_classifier
from acquisition.trainer import create_trainer
from acquisition.classifier_config import ClassifierConfig
import random
from acquisition.config import cache_dir
import os
import torch
from embed import load_model, extract_features_from_tokens
import math
from tqdm import tqdm

def get_wic_data(split):
    root_path = '/cs/labs/oabend/uriber/datasets/WiC'
    dir_path = os.path.join(root_path, split)

    sentences = []
    with open(os.path.join(dir_path, f'{split}.data.txt'), 'r') as fp:
        for line in fp:
            line_parts = line.strip().split('\t')
            target_word_inds = line_parts[2]
            target_word_ind1 = int(target_word_inds.split('-')[0])
            target_word_ind2 = int(target_word_inds.split('-')[1])
            sentences.append(((line_parts[3].split(), target_word_ind1), (line_parts[4].split(), target_word_ind2)))

    labels = []
    with open(os.path.join(dir_path, f'{split}.gold.txt'), 'r') as fp:
        for line in fp:
            if line.startswith('F'):
                labels.append(0)
            else:
                labels.append(1)

    return sentences, labels

def generate_features_for_wic(model_path, sentences):
    output_file_path = os.path.join(cache_dir, 'wid_features')
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
    sample_num = len(sentences)
    batch_num = math.ceil(sample_num/batch_size)

    checkpoint_len = 10
    for batch_ind in tqdm(range(first_batch, batch_num)):
        if batch_ind % checkpoint_len == 0:
            torch.save(res, output_file_path)
        batch_start = batch_ind * batch_size
        batch_end = min((batch_ind + 1) * batch_size, sample_num)
        batch = [sentences[i][0][0] for i in range(batch_start, batch_end)] + [sentences[i][1][0] for i in range(batch_start, batch_end)]
        cur_features = extract_features_from_tokens(batch, model, tokenizer, agg_subtokens_method='mean')
        for i in range(batch_size):
            vec1 = cur_features[i][sentences[batch_start+i][0][1]]
            vec2 = cur_features[batch_size+i][sentences[batch_start+i][1][1]]
            res.append(torch.cat([vec1, vec2]))
    print('Finished! Saving', flush=True)
    torch.save(res, output_file_path)

def get_data(model_path):
    sentences, labels = get_wic_data(split='test')
    features = generate_features_for_wic(model_path, sentences)

    assert len(features) == len(labels)

    # Features and pos data were created using different tokenizers, filter sentences that were tokenized differently
    data = []
    for feature_vector, label in zip(features, labels):
        if feature_vector is None:
            continue
        data.append((feature_vector, label))

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

def run_wsd_experiment(noise_images, version):
    config = ClassifierConfig()
    config.classifier_type = 'svm'
    os.remove(f'cache_dir/wic_features')
    noise_images_str = ''
    if noise_images:
        noise_images_str = '_noise_images'
    dir_path = f'result/mlm_itm_seed0{noise_images_str}/version_{version}/checkpoints'
    files_in_dir = os.listdir(dir_path)
    assert len(files_in_dir) == 1
    model_path = os.path.join(dir_path, files_in_dir[0])
    train_classifier(model_path, config)

from acquisition.classifier import create_classifier
from acquisition.trainer import create_trainer
from acquisition.classifier_config import ClassifierConfig
from acquisition.embed import load_model, embed_word
import random
import os
import torch
from tqdm import tqdm

def get_mcrae_data():
    tokens = []
    concreteness_list = []
    file_path = '/cs/labs/oabend/uriber/datasets/Concreteness_ratings_Brysbaert_et_al_BRM.txt'
    with open(file_path, 'r') as fp:
        first = True
        for line in fp:
            if first:
                first = False
                continue
            word, concreteness = collect_line(line)
            tokens.append(word)
            concreteness_list.append(concreteness)

    return tokens, concreteness_list

def collect_line(line):
    """ Each line in the input file is of the following format:
        <word> <bigram indicator> <concreteness mean> ... (some other not interesting columns)
        We care about the first three columns: the word, whether it's a bigram, and what is the concreteness of the
        word.
    """
    split_line = line.split()

    # The first number in each line is the bigram indicator: find it
    for token in split_line:
        if len(token) > 0 and not token[0].isalpha():
            break
    bigram_indicator = int(token)

    if bigram_indicator == 1:
        # It's a bigram: the first two tokens are the word
        word = split_line[0] + ' ' + split_line[1]
        concreteness = float(split_line[3])
    else:
        # It's a unigram: the word is only the first token
        word = split_line[0]
        concreteness = float(split_line[2])

    return word, concreteness

def generate_token_embeddings(model_path, tokens):
    res = []

    model, tokenizer = load_model(model_path)
    model.to(torch.device('cuda'))

    res = [embed_word(token, model, tokenizer, 'mean') for token in tqdm(tokens)]

    return res

def get_data(model_path):
    tokens, concreteness = get_mcrae_data()
    features = generate_token_embeddings(model_path, tokens)

    assert len(features) == len(concreteness)

    data = [(features[i], concreteness[i]) for i in range(len(concreteness))]

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

def run_concreteness_experiment(noise_images, version):
    config = ClassifierConfig()
    config.classifier_type = 'linear_regression'
    noise_images_str = ''
    if noise_images:
        noise_images_str = '_noise_images'
    dir_path = f'result/mlm_itm_seed0{noise_images_str}/version_{version}/checkpoints'
    files_in_dir = [x for x in os.listdir(dir_path) if x.startswith('epoch')]
    assert len(files_in_dir) == 1
    model_path = os.path.join(dir_path, files_in_dir[0])
    return train_classifier(model_path, config)

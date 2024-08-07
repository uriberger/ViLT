from acquisition.collect_flickr_data import collect_flickr_data
from acquisition.embed import generate_features
from acquisition.config import flickr_json_path, pos_tag_to_class, ontonotes_pos_tags
from acquisition.pos_tagging.generate_pos_data import generate_pos_data
from acquisition.classifier import create_classifier
from acquisition.trainer import create_trainer
from acquisition.classifier_config import ClassifierConfig
import random
import os
from datasets import load_dataset

def get_ontonotes_data(split, binary, pos_tag='noun'):
    if pos_tag == 'noun':
        class_ind = 0
    elif pos_tag == 'verb':
        class_ind = 1
    elif pos_tag == 'adjective':
        class_ind = 2

    dataset = load_dataset('conll2012_ontonotesv5', 'english_v12')
    token_lists = []
    pos_data = []
    for doc in dataset[split]:
        for sentence_obj in doc['sentences']:
            token_lists.append(sentence_obj['words'])
            if binary:
                pos_data.append([
                    {
                        'text': word,
                        'label': 1 if pos_tag_to_class[ontonotes_pos_tags[pos_tag]] == class_ind else 0
                    } for word, pos_tag in zip(sentence_obj['words'], sentence_obj['pos_tags'])
                ])
            else:
                pos_data.append([
                    {
                        'text': word,
                        'label': pos_tag_to_class[ontonotes_pos_tags[pos_tag]]
                    } for word, pos_tag in zip(sentence_obj['words'], sentence_obj['pos_tags'])
                ])

    return token_lists, pos_data

def get_data(model_path, binary, dataset, pos_tag='noun'):
    if dataset == 'flickr30k':
        sentences = collect_flickr_data(flickr_json_path, split='test')
        pos_data = generate_pos_data(sentences, binary, 'flickr30k', pos_tag=pos_tag)
        features = generate_features(model_path, 'flickr30k_features', sentences=sentences, tokens=None)
    elif dataset == 'ontonotes':
        tokens, pos_data = get_ontonotes_data(split='test', binary=binary, pos_tag=pos_tag)
        features = generate_features(model_path, 'ontonotes_features', sentences=None, tokens=tokens)

    assert len(features) == len(pos_data)

    # Features and pos data were created using different tokenizers, filter sentences that were tokenized differently
    data = []
    for feature_vectors, pos_data in zip(features, pos_data):
        if feature_vectors is None or feature_vectors.shape[0] != len(pos_data):
            continue
        data += [(feature_vectors[i], pos_data[i]['label'], pos_data[i]['text']) for i in range(len(pos_data))]

    random.shuffle(data)
    vocab = list(set([x[2].lower() for x in data]))
    train_words_num = int(0.8*len(vocab))
    test_words = set(vocab[train_words_num:])
    train_data = []
    test_data = []
    for feature_vector, pos_tag, word in data:
        if word.lower() in test_words:
            test_data.append((feature_vector, pos_tag))
        else:
            train_data.append((feature_vector, pos_tag))

    return train_data, test_data

def train_classifier(model_path, classifier_config, binary, dataset, pos_tag='noun'):
    train_data, test_data = get_data(model_path, binary, dataset, pos_tag=pos_tag)
    classifier = create_classifier(classifier_config)
    trainer = create_trainer(classifier, classifier_config, train_data, test_data)
    trainer.train()
    accuracy, res_mat = trainer.evaluate()
    return accuracy, res_mat

def run_pos_tagging_experiment(noise_images, version, dataset, pos_tag='noun'):
    config = ClassifierConfig()
    config.classifier_type = 'svm'
    if os.path.isfile(f'cache_dir/{dataset}_features'):
        os.remove(f'cache_dir/{dataset}_features')
    noise_images_str = ''
    if noise_images:
        noise_images_str = '_noise_images'
    dir_path = f'result/mlm_itm_seed0{noise_images_str}/version_{version}/checkpoints'
    files_in_dir = [x for x in os.listdir(dir_path) if x.startswith('epoch')]
    assert len(files_in_dir) == 1
    model_path = os.path.join(dir_path, files_in_dir[0])
    return train_classifier(model_path, config, True, dataset, pos_tag=pos_tag)

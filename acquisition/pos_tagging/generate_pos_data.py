import os
import json
from flair.data import Sentence
from flair.models import SequenceTagger
from acquisition.config import cache_dir
from tqdm import tqdm
from acquisition.config import pos_tag_to_class

def generate_pos_data(sentences, binary, filename, pos_tag='noun'):
    pos_tag2class_ind = {'noun': 0, 'verb': 1, 'adjective': 2, 'adverb': 3, 'function': 4, 'other': 5}
    if type(pos_tag) == str:
        class_inds = [pos_tag2class_ind[pos_tag]]
    elif type(pos_tag) == list:
        class_inds = [pos_tag2class_ind[x] for x in pos_tag]

    binary_str = '_binary' if binary else ''
    file_name = f'{filename}_pos_data{binary_str}.json'
    file_path = os.path.join(cache_dir, file_name)
    if os.path.isfile(file_path):
        with open(file_path, 'r') as fp:
            pos_data = json.load(fp)
    else:
        tagger = SequenceTagger.load("flair/pos-english")
        pos_data = []
        
        checkpoint_len = 100
        for count, sentence in tqdm(enumerate(sentences)):
            if count % checkpoint_len == 0:
                with open(file_path, 'w') as fp:
                    fp.write(json.dumps(pos_data))
            
            sentence_obj = Sentence(sentence)
            tagger.predict(sentence_obj)
            if binary:
                pos_data.append([
                    {
                        'text': token.text,
                        'start_position': token.start_position,
                        'label': 1 if pos_tag_to_class[token.annotation_layers['pos'][0]._value] in class_inds else 0
                    } for token in sentence_obj
                ])
            else:
                pos_data.append([
                    {
                        'text': token.text,
                        'start_position': token.start_position,
                        'label': pos_tag_to_class[token.annotation_layers['pos'][0]._value]
                    } for token in sentence_obj
                ])

        with open(file_path, 'w') as fp:
            fp.write(json.dumps(pos_data))

    return pos_data

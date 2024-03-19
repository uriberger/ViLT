import os
import json
from flair.data import Sentence
from flair.models import SequenceTagger
from acquisition.config import cache_dir
from tqdm import tqdm
from acquisition.config import class_to_pos_tag

pos_tag_to_class = {}
for class_ind in range(len(class_to_pos_tag)):
    for pos_tag in class_to_pos_tag[class_ind]:
        pos_tag_to_class[pos_tag] = class_ind

def generate_pos_data(sentences):
    file_name = 'flickr_pos_data.json'
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

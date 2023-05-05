import os
import json
from acquisition.config import cache_dir

def collect_flickr_data(root_path):
    file_name = 'flickr_sentences.json'
    file_path = os.path.join(cache_dir, file_name)
    if os.path.isfile(file_path):
        with open(file_path, 'r') as fp:
            sentences = json.load(fp)
    else:
        sentences = []
        tokens_file_path = os.path.join(root_path, 'tokens', 'results_20130124.token')
        with open(tokens_file_path, encoding='utf-8') as fp:
            for line in fp:
                split_line = line.strip().split('g#')
                caption = split_line[1].split('\t')[1]  # The first token is caption number
                sentences.append(caption)
        with open(file_path, 'w') as fp:
            fp.write(json.dumps(sentences))

    return sentences

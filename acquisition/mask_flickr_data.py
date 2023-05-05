import os
import json
import time
import stanza
from acquisition.config import cache_dir, flickr_root_path
from acquisition.collect_flickr_data import collect_flickr_data

def generate_nlp_data():
    file_name = 'flickr_nlp_data.json'
    file_path = os.path.join(cache_dir, file_name)
    if os.path.isfile(file_path):
        with open(file_path, 'r') as fp:
            nlp_data = json.load(fp)
    else:
        nlp = stanza.Pipeline('en')
        nlp_data = []
        sentences = collect_flickr_data(flickr_root_path)
        
        checkpoint_len = 100
        t = time.time()
        for sentence in sentences:
            count_so_far = len(nlp_data)
            if count_so_far % checkpoint_len == 0:
                print('Starting sample ' + str(count_so_far) + ' out of ' + str(len(sentences) + ', time from prev ' + str(time.time() - t)), flush=True)
                t = time.time()
                with open(file_path, 'w') as fp:
                    fp.write(json.dumps(nlp_data))
            cur_data = nlp(sentence)
            sentence_res = []
            for token in cur_data.sentences[0].tokens:
                token_dict = token.to_dict()[0]
                sentence_res.append({
                    'text': token_dict['text'],
                    'start_char': token_dict['start_char'],
                    'pos': token_dict['upos']
                })
            nlp_data.append(sentence_res)

        with open(file_path, 'w') as fp:
            fp.write(json.dumps(nlp_data))

    return nlp_data

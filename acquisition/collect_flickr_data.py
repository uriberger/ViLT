import json

def collect_flickr_data(json_path):
    with open(json_path, 'r') as fp:
        data = json.load(fp)['images']
    sentences = [x['raw'] for outer in data for x in outer['sentences']]

    return sentences

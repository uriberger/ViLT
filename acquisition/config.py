cache_dir = 'cache_dir'
flickr_root_path = '/cs/labs/oabend/uriber/datasets/flickr30'
mcrae_path = '/cs/labs/oabend/uriber/datasets/mcrae_typicality.yaml'
class_to_pos_tag = [
    # Nouns:
    ['NN', 'NNS', 'NNP', 'WP', 'NNPS', 'WP$'],
    # Verbs:
    ['VBD', 'VB', 'VBP', 'VBG', 'VBZ', 'VBN'],
    # Adjectivs:
    ['JJ', 'JJR', 'JJS'],
    # Others:
    ['<unk>', 'UH', ',', 'PRP', 'PRP$', 'RB', '.', 'DT', 'O', 'IN', 'CD', 'WRB', 'WDT',
     'CC', 'TO', 'MD', ':', 'RP', 'EX', 'FW', 'XX', 'HYPH', 'POS', 'RBR', 'PDT', 'RBS',
     'AFX', '-LRB-', '-RRB-', '``', "''", 'LS', '$', 'SYM', 'ADD']
]
embed_dim = 768
subword_pooling = 'mean' # mean/last

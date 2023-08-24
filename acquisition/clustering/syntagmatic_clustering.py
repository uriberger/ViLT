from collections import defaultdict
from acquisition.clustering.clustering_utils import create_clusters
from acquisition.config import swow_path
import os
from csv import reader

def build_swow_dataset(word_list):
    word_dict = {x: True for x in word_list}
    strength_dict = {}

    with open(swow_path, newline='', encoding='utf8') as csvfile:
        swow_reader = reader(csvfile, delimiter='\t')
        for row in swow_reader:
            ''' Words might appear in different orders. For example, 'dog' could be the cue and the participant
            responded using 'cat', but the other way around is also possible. We're not interested in the order so
            we we keep a single entry for each pair (the first word would be the first in alphabetical order).
            When there are multiple entries will sum the strength of all entries. '''
            word1 = sorted([row[1], row[2]])[0]
            word2 = sorted([row[1], row[2]])[1]
            strength = row[3]
            if word1 in word_dict and word2 in word_dict:
                if word1 not in strength_dict:
                    strength_dict[word1] = {}
                if word2 not in strength_dict[word1]:
                    strength_dict[word1][word2] = 0
                    strength_dict[word1][word2] += int(strength)

    return strength_dict


def evaluate_clusters(assoc_strength_dataset, word_to_cluster):
    cluster_to_word_list = defaultdict(list)
    for word, cluster_ind in word_to_cluster.items():
        cluster_to_word_list[cluster_ind].append(word)
    cluster_pair_lists = [
        [x for outer in [[(z[i], z[j]) for j in range(i + 1, len(z))] for i in range(len(z))] for x in outer] for z
        in cluster_to_word_list.values()]
    all_pair_lists = [x for outer in cluster_pair_lists for x in outer]

    strength_sum = 0
    found = 0
    not_found = 0
    for x in all_pair_lists:
        word1 = sorted([x[0], x[1]])[0]
        word2 = sorted([x[0], x[1]])[1]
        if word1 in assoc_strength_dataset and word2 in assoc_strength_dataset[word1]:
            strength_sum += assoc_strength_dataset[word1][word2]
            found += 1
        else:
            not_found += 1
    return strength_sum / found, found, not_found

def evaluate_clustering(model_path, word_list, cluster_num=100):
    cluster_list = create_clusters(model_path, word_list, cluster_num)
    word_to_cluster = {word_list[i]: cluster_list[i] for i in range(len(word_list))}
    swow_dataset = build_swow_dataset(word_list)
    return evaluate_clusters(swow_dataset, word_to_cluster)

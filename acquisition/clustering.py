import yaml
import torch
from acquisition.config import mcrae_path, subword_pooling
from embed import load_model, extract_embeddings
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score

def build_category_dataset(input_file_path):
    with open(input_file_path) as f:
        # First, create a dictionary of category: typicality rating for each word
        base_category_to_word = yaml.load(f, Loader=yaml.FullLoader)
        word_lists = [list(x.keys()) for x in base_category_to_word.values()]
        all_words = list(set([word for outer in word_lists for word in outer]))
        word_to_categories = {word: {
            x[0]: x[1][word] for x in base_category_to_word.items() if word in x[1]
        } for word in all_words}

        # Now, for each word, take the category with the highest typicality rating
        word_to_category = {x[0]: max(x[1], key=x[1].get) for x in word_to_categories.items()}

        # Finally, create the reversed mapping: categories to a list of words
        all_categories = list(word_to_category.values())
        category_to_word_list = {categ: [x[0] for x in word_to_category.items() if x[1] == categ]
                                    for categ in all_categories}

        return category_to_word_list

def create_clusters(model_path):
    category_to_word_list = build_category_dataset(mcrae_path)
    model, tokenizer = load_model(model_path)
    model.eval()

    # Create embeddings
    all_words = [x for outer in list(category_to_word_list.values()) for x in outer]
    with torch.no_grad():
        embeddings = extract_embeddings(all_words, model, tokenizer)
        embeddings = embeddings[:, 1:-1, :] # Remove cls and sep tokens
        if subword_pooling == 'mean':
            embeddings = embeddings.mean(dim=1)
        elif subword_pooling == 'last':
            embeddings = embeddings[:, -1, :]

    kmeans = KMeans(n_clusters=41).fit(embeddings)
    cluster_list = list(kmeans.labels_)

    word_to_category = {}
    for category, word_list in category_to_word_list.items():
        for word in word_list:
            word_to_category[word] = category

    return word_list, [word_to_category[x] for x in word_list], cluster_list

def aggregate_intersection_counts(gt_labels, predicted_labels):
    cluster_to_gt_intersection = {}
    gt_to_cluster_intersection = {}
    gt_class_count = {}
    cluster_count = {}
    for i in range(len(gt_labels)):
        gt_class = gt_labels[i]
        predicted_cluster = predicted_labels[i]

        # Update counts
        if gt_class not in gt_class_count:
            gt_class_count[gt_class] = 0
        gt_class_count[gt_class] += 1
        if predicted_cluster not in cluster_count:
            cluster_count[predicted_cluster] = 0
        cluster_count[predicted_cluster] += 1

        # Update gt class to cluster mapping
        if gt_class not in gt_to_cluster_intersection:
            gt_to_cluster_intersection[gt_class] = {predicted_cluster: 0}
        if predicted_cluster not in gt_to_cluster_intersection[gt_class]:
            gt_to_cluster_intersection[gt_class][predicted_cluster] = 0
        gt_to_cluster_intersection[gt_class][predicted_cluster] += 1

        # Update cluster to gt class mapping
        if predicted_cluster not in cluster_to_gt_intersection:
            cluster_to_gt_intersection[predicted_cluster] = {gt_class: 0}
        if gt_class not in cluster_to_gt_intersection[predicted_cluster]:
            cluster_to_gt_intersection[predicted_cluster][gt_class] = 0
        cluster_to_gt_intersection[predicted_cluster][gt_class] += 1

    return cluster_to_gt_intersection, gt_to_cluster_intersection, gt_class_count, cluster_count

def calc_purity_collocation(cluster_to_gt_intersection, gt_to_cluster_intersection):
    N = sum([sum(x.values()) for x in gt_to_cluster_intersection.values()])
    if N == 0:
        return 0, 0, 0

    purity = (1 / N) * sum([max(x.values()) for x in cluster_to_gt_intersection.values()])
    collocation = (1 / N) * sum([max(x.values()) for x in gt_to_cluster_intersection.values()])
    if purity + collocation == 0:
        f1 = 0
    else:
        f1 = 2 * (purity * collocation) / (purity + collocation)

    return purity, collocation, f1

def calc_fscore(gt_to_cluster_intersection, gt_class_count, cluster_count):
    N = sum([sum(x.values()) for x in gt_to_cluster_intersection.values()])
    if N == 0:
        return 0

    solution_Fscore = 0
    # Go over all gt classes and calculate class_Fscore for each class
    for gt_class, cluster_map in gt_to_cluster_intersection.items():
        gt_class_size = gt_class_count[gt_class]
        class_Fscore = 0
        # Go over all the clusters and calculate f value for each cluster, with the current class
        for cluster, intersection_size in cluster_map.items():
            cluster_size = cluster_count[cluster]
            precision = intersection_size / gt_class_size
            recall = intersection_size / cluster_size
            if precision + recall == 0:
                f_value = 0
            else:
                f_value = 2 * (precision * recall) / (precision + recall)
            if f_value > class_Fscore:
                class_Fscore = f_value

        solution_Fscore += (gt_class_size / N) * class_Fscore

    return solution_Fscore

def evaluate_clusters(gt_labels, predicted_labels):
    cluster_to_gt_intersection, gt_to_cluster_intersection, gt_class_count, cluster_count = \
        aggregate_intersection_counts(gt_labels, predicted_labels)

    v_measure = v_measure_score(gt_labels, predicted_labels)
    purity, collocation, pu_co_f1 = calc_purity_collocation(cluster_to_gt_intersection, gt_to_cluster_intersection)
    fscore = calc_fscore(gt_to_cluster_intersection, gt_class_count, cluster_count)

    return {
        'v_measure_score': v_measure,
        'purity': purity,
        'collocation': collocation,
        'pu_co_f1': pu_co_f1,
        'fscore': fscore
    }

def evaluate_clustering(model_path):
    word_list, gt_labels, predicted_labels = create_clusters(model_path)
    results = evaluate_clustering(gt_labels, predicted_labels)

    return results

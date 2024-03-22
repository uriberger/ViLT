from acquisition.embed import load_model, extract_embeddings
import torch
from acquisition.config import subword_pooling
from sklearn.cluster import KMeans

def create_clusters(model_path, word_list, cluster_num):
    model, tokenizer = load_model(model_path)
    model.eval()

    # Create embeddings
    with torch.no_grad():
        embeddings = extract_embeddings(word_list, model, tokenizer)
        embeddings = [x[1:-1, :] for x in embeddings] # Remove cls and sep tokens
        if subword_pooling == 'mean':
            embeddings = [x.mean(dim=0) for x in embeddings]
        elif subword_pooling == 'last':
            embeddings = [x[-1, :] for x in embeddings]
        embeddings = [x.unsqueeze(dim=0) for x in embeddings]
        embeddings = torch.cat(embeddings, dim=0)

    kmeans = KMeans(n_clusters=cluster_num).fit(embeddings)
    cluster_list = list(kmeans.labels_)

    return cluster_list
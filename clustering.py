import warnings; warnings.filterwarnings('ignore')
import os
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
import bcubed

def list_pt_files(directory):
    audio_exts = {".pt", ".pth", ".bin"}
    audio_files = []

    for root, _, files in os.walk(directory):
        for f in files:
            if os.path.splitext(f)[1].lower() in audio_exts:
                audio_files.append(os.path.join(root, f))
    return audio_files


if __name__ == "__main__":
    DIRECTORY = "mos_embeddings"
    pt_files = list_pt_files(DIRECTORY)
    model_stat_dict = {} 
    
    reference = []
    embeddings = []
    model_names = set()
    for pt_file in pt_files:
        model_name = "".join(os.path.basename(pt_file).split("-")[:-1])
        model_names.add(model_name)
        reference.append(model_name)
        result = torch.load(pt_file)
        embeddings.append(result['embedding'].numpy())

    labels = AgglomerativeClustering(n_clusters=len(model_names), metric='cosine', linkage='average').fit(np.array(embeddings)).labels_

    reference = {i: {reference[i]} for i in range(len(reference))}
    hypothesis = {i: {labels[i]} for i in range(len(labels))}
    
    precision = bcubed.precision(hypothesis, reference)
    recall = bcubed.recall(hypothesis, reference)
    f1 = bcubed.fscore(precision, recall)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize
    
    X = normalize(embeddings, norm="l2")
    kmeans = KMeans(
        n_clusters=len(model_names),
        init="k-means++",
        n_init=32,
        max_iter=512,
        random_state=0,
    )

    labels = kmeans.fit_predict(X)

    hypothesis = {i: {labels[i]} for i in range(len(labels))}
    
    precision = bcubed.precision(hypothesis, reference)
    recall = bcubed.recall(hypothesis, reference)
    f1 = bcubed.fscore(precision, recall)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
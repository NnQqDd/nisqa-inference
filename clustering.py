import warnings; warnings.filterwarnings('ignore')
import os
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize    
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
    # DIRECTORY = "voice-clone-audios-metrics-embeddings"
    DIRECTORY = "voice-conversion-audios-metrics-embeddings"
    pt_files = list_pt_files(DIRECTORY)
    model_stat_dict = {} 
    
    labels = []
    embeddings = []
    model_names = set()
    except_names = {}
    for pt_file in pt_files:
        model_name = os.path.basename(os.path.dirname(pt_file))
        if model_name in except_names:
            continue
        model_names.add(model_name)
        labels.append(model_name)
        result = torch.load(pt_file)
        if len(result['embedding'].shape) == 2 and result['embedding'].shape[0] != 1:
            embeddings.append(result['embedding'][0].numpy())
        else:
            embeddings.append(result['embedding'].squeeze().numpy())

    print("Models     :", model_names)
    print("Embeddings :", len(embeddings))
    embeddings = np.array(embeddings)

    X = normalize(embeddings, norm="l2")
    # print(X.shape)
    kmeans = KMeans(
        n_clusters=len(model_names),
        init="k-means++",
        n_init=64,
        max_iter=512,
        random_state=0,
    )
    preds = kmeans.fit_predict(X)

    reference = {i: {labels[i]} for i in range(len(labels))}
    hypothesis = {i: {preds[i]} for i in range(len(preds))}
    
    precision = bcubed.precision(hypothesis, reference)
    recall = bcubed.recall(hypothesis, reference)
    f1 = bcubed.fscore(precision, recall)
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1       : {f1:.3f}")


    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import LabelEncoder

    tsne = TSNE(n_components=2, perplexity=50, random_state=42)
    numeric_labels = LabelEncoder().fit_transform(labels)
    embeddings_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        c=numeric_labels,          # <--- THIS IS THE KEY
        s=10, 
        alpha=0.6, 
        cmap='tab20'
    )

    # 3. Add a colorbar that actually shows the 6 clusters
    n_labels = len(model_names)
    cb = plt.colorbar(scatter, ticks=range(n_labels))
    cb.set_label('Cluster ID')

    plt.title(f"t-SNE: {n_labels} Cluster(s) Visualized")
    plt.savefig(f'{DIRECTORY}-tsne.png', dpi=300, bbox_inches='tight')
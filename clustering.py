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
    DIRECTORY = "voice-conversion-audios-mos-embeddings"
    pt_files = list_pt_files(DIRECTORY)
    model_stat_dict = {} 
    
    reference = []
    embeddings = []
    model_names = set()
    except_names = {
        "tortoise_tts_fast", "tortoise_tts_ultra_fast", 'f5tts_v1_base_16', 'f5tts_v1_base_64', 'e2tts_base_16', 'e2tts_base_64'
    }
    for pt_file in pt_files:
        model_name = "".join(os.path.basename(pt_file).split("-")[:-1])
        if model_name in except_names:
            continue
        model_names.add(model_name)
        reference.append(model_name)
        result = torch.load(pt_file)
        embeddings.append(result['embedding'].numpy())

    print("Models     :", model_names)
    print("Embeddings :", len(embeddings))

    X = normalize(embeddings, norm="l2")
    kmeans = KMeans(
        n_clusters=len(model_names),
        init="k-means++",
        n_init=64,
        max_iter=512,
        random_state=0,
    )
    labels = kmeans.fit_predict(X)

    reference = {i: {reference[i]} for i in range(len(reference))}
    hypothesis = {i: {labels[i]} for i in range(len(labels))}
    
    precision = bcubed.precision(hypothesis, reference)
    recall = bcubed.recall(hypothesis, reference)
    f1 = bcubed.fscore(precision, recall)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
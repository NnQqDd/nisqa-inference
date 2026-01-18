import warnings; warnings.filterwarnings('ignore')
import os
import torch
from pprint import pprint


def list_pt_files(directory):
    audio_exts = {".pt", ".pth", ".bin"}
    audio_files = []

    for root, _, files in os.walk(directory):
        for f in files:
            if os.path.splitext(f)[1].lower() in audio_exts:
                audio_files.append(os.path.join(root, f))
    return audio_files


if __name__ == "__main__":
    DIRECTORY = "tts_embeddings"
    pt_files = list_pt_files(DIRECTORY)
    model_stat_dict = {} 
    
    for pt_file in pt_files:
        model_name = "".join(os.path.basename(pt_file).split("-")[:-1])
        if model_name not in model_stat_dict:
            model_stat_dict[model_name] =     {
                "n_embeddings": 0,
                "avg_score": 0,
                "std_score": 0,
                "min_score": 1e9,
                "max_score": 0,
            }
        result = torch.load(pt_file)
        score = result["score"]
        embedding = result["embedding"]
        stat_dict = model_stat_dict[model_name]
        stat_dict["n_embeddings"] += 1
        stat_dict["avg_score"] += score
        stat_dict["std_score"] += score**2
        stat_dict["min_score"] = min(stat_dict["min_score"], score)
        stat_dict["max_score"] = max(stat_dict["max_score"], score)
    
    for model_name, stat_dict in model_stat_dict.items():
        stat_dict["avg_score"] /= stat_dict["n_embeddings"]
        stat_dict["std_score"] /= stat_dict["n_embeddings"]
        stat_dict["std_score"] -= stat_dict["avg_score"]**2
        stat_dict["std_score"] = stat_dict["std_score"]**0.5
    
    pprint(model_stat_dict, sort_dicts=False)
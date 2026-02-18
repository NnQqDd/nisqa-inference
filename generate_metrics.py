import warnings; warnings.filterwarnings('ignore')
import os
from pathlib import Path
import torch
from tqdm import tqdm
from nisqa.NISQA_model import nisqaModel


def list_audio_files(directory):
    audio_exts = {".wav", ".mp3", ".flac"}
    audio_files = []

    for root, _, files in os.walk(directory):
        for f in files:
            if os.path.splitext(f)[1].lower() in audio_exts:
                audio_files.append(os.path.join(root, f))
    return audio_files

if __name__ == "__main__":
    args = dict()
    args['pretrained_model'] = './weights/nisqa.tar'
    args['deg'] = args['output_dir'] = ''
    args['mode'] = 'predict_file'
    args['tr_bs_val'] = args['tr_num_workers'] = args['ms_channel'] = 1
    args['ms_sr'] = None

    # PATH = "/home/duyn/ActableDuy/voice-synthesis/voice-clone-audios"
    PATH = "/home/duyn/ActableDuy/voice-synthesis/voice-conversion-audios"
    nisqa = nisqaModel(args)
    os.makedirs(f"{PATH.split('/')[-1]}-metrics-embeddings", exist_ok=True)
    audio_paths = list_audio_files(PATH)
    ignore_names = {}
    audio_paths = [audio_path for audio_path in audio_paths if os.path.basename(os.path.dirname(audio_path)) not in ignore_names]
    for audio_path in tqdm(audio_paths):
        nisqa.args['deg'] = audio_path
        nisqa._loadDatasetsFile()
        pred, embedding = nisqa.predict()
        result = {
            "embedding": embedding.cpu(),
            "score": pred[0],
        }
        name = os.path.basename(audio_path).split(".")[0] + ".pt"
        dirname = os.path.basename(os.path.dirname(audio_path))
        os.makedirs(f"{PATH.split('/')[-1]}-metrics-embeddings/{dirname}", exist_ok=True)
        torch.save(result, 
            f"{PATH.split('/')[-1]}-metrics-embeddings/{dirname}/{name}"
        )
    
        
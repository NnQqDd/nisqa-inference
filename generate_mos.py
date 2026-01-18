import warnings; warnings.filterwarnings('ignore')
import os
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
    args['pretrained_model'] = './weights/nisqa_mos_only.tar'
    args['deg'] = args['output_dir'] = ''
    args['mode'] = 'predict_file'
    args['tr_bs_val'] = args['tr_num_workers'] = args['ms_channel'] = 1
    args['ms_sr'] = None

        
    nisqa = nisqaModel(args)
    os.makedirs("mos_embeddings", exist_ok=True)
    audio_paths = list_audio_files("/home/duyn/ActableDuy/voice-synthesis/synthetic-audios")
    for audio_path in tqdm(audio_paths):
        nisqa.args['deg'] = audio_path
        nisqa._loadDatasetsFile()
        pred, embedding = nisqa.predict()
        result = {
            "embedding": embedding.cpu(),
            "score": pred.item(),
        }
        torch.save(result, f"mos_embeddings/{os.path.basename(audio_path)}.pt")
    
        
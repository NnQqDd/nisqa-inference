import warnings; warnings.filterwarnings('ignore')
from nisqa.NISQA_model import nisqaModel

if __name__ == "__main__":
    print('-----------------------------------------------------------------------')
    args = dict()
    args['pretrained_model'] = '/home/duyn/ActableDuy/NISQA/weights/nisqa.tar'
    args['deg'] = 'reference.flac'
    args['output_dir'] = 'results'
    args['mode'] = 'predict_file'
    args['tr_bs_val'] = 1
    args['tr_num_workers'] = 1
    args['ms_channel'] = 1
    args['ms_sr'] = None
        
    nisqa = nisqaModel(args)
    pred, embeddings = nisqa.predict()
    print('AUDIO PATH     :', args['deg'])
    print('[MOS, NOISE, DISCONTINUATION, COLORATION, LOUDNESS]:', pred)
    print('EMBEDDING SHAPE:', embeddings.shape)

if __name__ == "__main__":
    print('-----------------------------------------------------------------------')
    args = dict()
    args['pretrained_model'] = '/home/duyn/ActableDuy/NISQA/weights/nisqa_mos_only.tar'
    args['deg'] = 'reference.flac'
    args['output_dir'] = 'results'
    args['mode'] = 'predict_file'
    args['tr_bs_val'] = 1
    args['tr_num_workers'] = 1
    args['ms_channel'] = 1
    args['ms_sr'] = None
        
    nisqa = nisqaModel(args)
    pred, embedding = nisqa.predict()
    print('AUDIO PATH     :', args['deg'])
    print('MOS            :', pred)
    print('EMBEDDING SHAPE:', embedding.shape)


if __name__ == "__main__":
    print('-----------------------------------------------------------------------')
    args = dict()
    args['pretrained_model'] = '/home/duyn/ActableDuy/NISQA/weights/nisqa_tts.tar'
    args['deg'] = 'reference.flac'
    args['output_dir'] = 'results'
    args['mode'] = 'predict_file'
    args['tr_bs_val'] = 1
    args['tr_num_workers'] = 1
    args['ms_channel'] = 1
    args['ms_sr'] = None
        
    nisqa = nisqaModel(args)
    pred, embedding = nisqa.predict()
    print('AUDIO PATH     :', args['deg'])
    print('MOS            :', pred)
    print('EMBEDDING SHAPE:', embedding.shape)
























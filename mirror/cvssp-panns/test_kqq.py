import numpy as np
import torch
from panns_inference import labels

import panns_hear

import librosa

def print_audio_tagging_result(clipwise_output):
    """Visualization of audio tagging result.
    Args:
      clipwise_output: (classes_num,)
    """
    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # Print audio tagging top probabilities
    for k in range(10):
        print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], 
            clipwise_output[sorted_indexes[k]]))


audio_path = 'resources/resources_R9_ZSCveAHg_7s.mp3'
audio, fs = librosa.load(audio_path, sr=32000, mono=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = panns_hear.load_model('Cnn14_mAP=0.431.pth', device)


x = torch.Tensor(audio[None, :]).to(device)

with torch.no_grad():
    model.eval()
    clipwise_output = model(x)['clipwise_output'][0]
    clipwise_output = clipwise_output.data.cpu().numpy()
    # from IPython import embed; embed(using=False); os._exit(0)
    
print_audio_tagging_result(clipwise_output)
from IPython import embed; embed(using=False); os._exit(0)


x = torch.Tensor(audio[None, :]).to(device)
z_s = panns_hear.get_scene_embeddings(x, model)
z_t = panns_hear.get_timestamp_embeddings(x, model)
print(z_s.shape)
print(z_t[0].shape)
print(z_t[1].shape)

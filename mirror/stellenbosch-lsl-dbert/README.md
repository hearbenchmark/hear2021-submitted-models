# audio_dbert
A general audio embedding model for the HEAR 2021 challenge. 
In short, the model uses a modified BERT transformer and CNN encoder (like wav2vec 2), with a special head and loss formulation to learn from unlabeled audio.
Training is performed as a GAN -- with the main BERT audio encoder acting as the discriminator model, and several clustering models (not used during inference) acting as the generator model.

## Installation
The package is installed with `python -m pip install git+https://github.com/RF5/audio_dbert.git` . 

If that does not work out of the box, it might mean that you don't have the right cuda drivers of some kind of another.
If CUDA and cuDNN versions are giving one problems, then consider installing pytorch and associated CUDA drivers seperately.
Concretely, here are the versions of pytorch and cuda toolkit (as conda packages) that should work:
- `torch==1.9.1` ; it should also work with torch 1.10, but no guarantees. 
- `cudatoolkit=11.3.1`
- `cudnn=8.2.1=cuda11.3_0`

Other packages are much simpler and should be installed automatically during package installation.

## Trained model weights
The trained weights are available as a github release on this repository. Namely, [this link](https://github.com/RF5/audio_dbert/releases/download/v0.5/e105-2_checkpoint_4_920000_Diter184001.pt) provides the checkpoint for the trained model that should work with the `load_model` function.


You can also change which layer (from 1-24) to use features from (16 by default, `None` to use final layer after projection). This is changed by the optional `layer` argument in `load_model`.

## A note on very long audio samples
To fulfill the 16GB memory limit for audio samples up to 20min, inference is chunked into `m_chunk` minute segments and then combined. From internal testing, chunking at 4 minutes seems to keep the memory to within 16GB for 20 min audio and a batch size of 1. If for some reason you are exceeding this, please lower the `m_chunk` optional argument for `get_timestamp_embeddings` and `get_scene_embeddings` -- this is the number of minutes to chunk the audio into when doing inference, currently set at a default of 4.

## Questions & Troubleshooting
If there are any questions or problems, please raise a github issue on this and I'll attend to it as soon as possible.

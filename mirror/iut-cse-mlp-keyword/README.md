# Audio-MLP

MLP-based models for learning audio representations. Submission for [HEAR-2021@NeurIPS'21](https://neuralaudio.ai/hear2021-holistic-evaluation-of-audio-representations.html).

## Setup

```
pip install -e 'git+https://github.com/ID56/HEAR-2021-Audio-MLP.git#egg=hearaudiomlp'
```

## Usage

```python
from hearaudiomlp.kwmlp import load_model, get_timestamp_embeddings, get_scene_embeddings # or from hearaudiomlp.audiomlp

model = load_model("checkpoints/kwmlp.pth")

b, ms, sr = 2, 1000, 16000
dummy_input = torch.randn(b, int(sr * ms / 1000))

embeddings, timestamps = get_timestamp_embeddings(dummy_input, model)
scene_embeddings = get_scene_embeddings(dummy_input, model)
```

## Models

|   Model Name    | # Params† | GFLOPS*† | Sampling Rate | Hop Length | Timestamp Embedding | Scene Embedding |  Location     |
| --------------- | --------- | -------  | ------------- | ---------- | ------------------- | --------------- | ------------- |
|     kwmlp       |    424K   | 0.034    |    16000      |    10ms    |  64                 |   1024          |  [kwmlp(1.7Mb)](checkpoints/kwmlp.pth)   |
|    audiomae     |    213K   | 0.023    |    16000      |    10ms    |  8                  |   1584          |  [audiomae(0.9Mb)](checkpoints/audiomae.pth)   |

† <sub>Only considering the encoder, which is used for generating embeddings.</sub><br>
\* <sub>Although there is no direct way to count FLOPS like parameters, you can use [facebookresearch/fvcore](https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md). The FLOPS measured are per 1 single input spectrogram (tensor of shape `(1, 40, 98)`).</sub>

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/80x15.png" /></a><br />The trained checkpoints are licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>, as per HEAR-2021 requirements. You may also download them from drive: [ [kwmlp](https://drive.google.com/uc?id=1-49LCU_zJODhmaXXJnjzsfr0ukCtHzRg&export=download) | [audiomae](https://drive.google.com/uc?id=16b96Ske0yhHE99U708lzQ_ob5KxHukiP&export=download) ].

## Notes

All models were trained on:
- A standard Kaggle environment: a single 16GiB NVIDIA Tesla P100, CUDA 11.0, CuDNN 8.0.5, python 3.7.10.
- KW-MLP was trained on Google Speech Commands V2-35, and the weights are a direct port of its paper [1].
- AudioMAE is an adaptation of KW-MLP that was trained on the training splits from the HEAR2021 Open tasks.
- Both models primarily utilize gated-MLPs [2].

## References

```bibtex
@misc{morshed2021attentionfree,
      title   = {Attention-Free Keyword Spotting}, 
      author  = {Mashrur M. Morshed and Ahmad Omar Ahsan},
      year    = {2021},
      eprint  = {2110.07749},
      archivePrefix = {arXiv},
      primaryClass  = {cs.LG}
}
```

```bibtex
@misc{liu2021pay,
      title  = {Pay Attention to MLPs}, 
      author = {Hanxiao Liu and Zihang Dai and David R. So and Quoc V. Le},
      year   = {2021},
      eprint = {2105.08050},
      archivePrefix = {arXiv},
      primaryClass  = {cs.LG}
}
```
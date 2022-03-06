# HEAR 2021 Submission


### Installation

```python
pip install git+https://github.com/KinWaiCheuk/my_package.git
```

### Usage

Audio embeddings can be computed using one of two methods: 1)
`get_scene_embeddings`, or 2) `get_timestamp_embeddings`.

`get_scene_embeddings` accepts a batch of audio clips and produces a single embedding
for each audio clip. This can be computed like so:
```python
import torch
import hearbaseline

# Load model with weights - located in the root directory of this repo
model = hearbaseline.load_model("saved_models/naive_baseline.pt")

# Create a batch of 2 white noise clips that are 2-seconds long
# and compute scene embeddings for each clip
audio = torch.rand((2, model.sample_rate * 2))
embeddings = hearbaseline.get_scene_embeddings(audio, model)
```

The `get_timestamp_embeddings` method works exactly the same but returns an array
of embeddings computed every 25ms over the duration of the input audio. An array
of timestamps corresponding to each embedding is also returned.

See the [common API](https://neuralaudio.ai/hear2021-holistic-evaluation-of-audio-representations.html#common-api)
for more details.

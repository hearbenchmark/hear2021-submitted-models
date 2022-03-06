# HEAR 2021 Baseline
I used a Dockerfile-cuda10.2, developed with hear-eval-kit.
CUDA:10.2


## Installation
**pip local source tree**
```python
git clone https://github.com/ibkuroyagi/hear2021-submit.git
cd hearline
pip install -e .
```

## demonstration
1. Put model in ./saved_models/checkpoint-2821steps.pkl

```
hear-validator hearline --model ./saved_models/checkpoint-2821steps.pkl
```

team name: ibkuroyagi
module name: hearline
model's URL: https://github.com/ibkuroyagi/hear2021-submit/blob/main/saved_models/checkpoint-2821steps.pkl
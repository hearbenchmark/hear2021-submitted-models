import torch
from hearline import get_timestamp_embeddings
import hearline

# Load model with weights - located in the root directory of this repo
device = "cuda"
model = hearline.load_model("saved_models/checkpoint-2821steps.pkl")

# Create a batch of 2 white noise clips that are 2-seconds long
# and compute scene embeddings for each clip
audio = torch.rand((2, model.sample_rate * 60)).to(device)
embeddings = hearline.get_scene_embeddings(audio, model)
print(embeddings.shape)
time_embeddings, timestamps = get_timestamp_embeddings(audio, model)
print(time_embeddings.shape, timestamps.shape)

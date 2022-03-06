
"""
HEAR2021 API implementation

As per specifications in
https://neuralaudio.ai/hear2021-holistic-evaluation-of-audio-representations.html
"""

HOP_SIZE_TIMESTAMPS = 0.050 # <50 ms recommended
HOP_SIZE_SCENE = 0.5

import time

import openl3
import numpy
import structlog
import tensorflow as tf

#import tensorflow_datasets
#from tensorflow_datasets.typing import Tensor
#from tensorflow.types.experimental import Tensor
from typing import NewType, Tuple
Tensor = NewType('Tensor', object)

log = structlog.get_logger()


class Model(tf.Module):
    def __init__(self, model, sample_rate=48000, embedding_size=512):
        self.sample_rate = sample_rate
        self.scene_embedding_size = embedding_size
        self.timestamp_embedding_size = embedding_size

        self.openl3_model = model # the OpenL3 model instance    


def load_model(model_file_path: str) -> Model:
    # FIXME: respect model_file_path

    embedding_size = 512

    openl3_model = openl3.models.load_audio_embedding_model(input_repr="mel256",
                            content_type="music",
                            embedding_size=embedding_size,
    )

    model = Model(model=openl3_model, embedding_size=embedding_size)
    return model

TimestampedEmbeddings = Tuple[Tensor, Tensor]

def get_timestamp_embeddings(
    audio: Tensor,
    model: Model,
    hop_size=HOP_SIZE_TIMESTAMPS,
) -> TimestampedEmbeddings:
    """
    audio: n_sounds x n_samples of mono audio in the range [-1, 1]
    model: Loaded Model. 

    Returns:

        embedding: A float32 Tensor with shape (n_sounds, n_timestamp, model.timestamp_embedding_size).
        timestamps: Tensor. Centered timestamps in milliseconds corresponding to each embedding in the output.
     """
    # pre-conditions
    assert len(audio.shape) == 2

    # get embeddings for a single audio clip
    def get_embedding(samples):
        emb, ts = openl3.get_audio_embedding(samples,
            sr=model.sample_rate,
            model=model.openl3_model,
            hop_size=hop_size,
            center=True,
            verbose=0,
        )

        return emb, ts

    # Compute embeddings for each clip
    embeddings = []
    timestamps = []

    # convert to Numpy
    pre_convert_start = time.time()
    samples = numpy.array(audio)
    pre_convert_end = time.time()

    input_sample_length = audio.shape[1]/model.sample_rate

    # pad audio with silence
    # ensures that events at end of input can be caught
    pad_samples = int(((1.0/2.0)-hop_size) * model.sample_rate)
    samples = numpy.pad(samples,
            pad_width=[(0, 0), (0, pad_samples)],
            mode='constant', constant_values=0,
    )

    compute_start = time.time()
    for sound_no in range(audio.shape[0]):
        emb, ts = get_embedding(samples[sound_no, :])
        embeddings.append(emb)
        # HEAR timestamps are in milliseconds
        ts = ts * 1000.0
        timestamps.append(ts)
    compute_end = time.time()

    # convert to Tensorflow
    post_convert_start = time.time()
    emb = numpy.stack(embeddings)
    ts = numpy.stack(timestamps)
    emb = tf.convert_to_tensor(emb)
    ts = tf.convert_to_tensor(ts)
    post_convert_end = time.time()

    # post-conditions
    assert len(ts.shape) == 2 
    assert len(ts) >= 1
    assert emb.shape[0] == audio.shape[0]
    assert len(emb.shape) == 3, emb.shape
    assert ts.shape[0] == audio.shape[0]
    assert emb.shape[1] == ts.shape[1], (emb.shape, ts.shape)
    assert emb.shape[2] == model.timestamp_embedding_size
    if len(ts) >= 2:
        assert ts[0,0] >= 0.0, ts
        assert ts[0,-1] <= (input_sample_length*1000.0), ts
        assert ts[0,1] == ts[0,0] + (hop_size*1000.0), ts


    log.debug('get-timestamp-embeddings',
        n_samples=audio.shape[0],
        sample_length=input_sample_length,
        pre_convert_duration=pre_convert_end-pre_convert_start,
        post_convert_duration=post_convert_end-post_convert_start,
        compute_duration=compute_end-compute_start,
    )

    return (emb, ts)


def get_scene_embeddings(
    audio: Tensor,
    model: Model,
    hop_size=HOP_SIZE_SCENE,
) -> Tensor:

    """
    audio: n_sounds x n_samples of mono audio in the range [-1, 1].
    model: Loaded Model.

    Returns:

        embedding: A float32 Tensor with shape (n_sounds, model.scene_embedding_size).
    """
    assert len(audio.shape) == 2 

    embeddings, ts = get_timestamp_embeddings(audio, model, hop_size=hop_size)

    # FIXME: use TensorFlow Tensor instead. Using tf.constant ?
    emb = numpy.mean(embeddings, axis=1)
    emb = tf.convert_to_tensor(emb)

    assert len(emb.shape) == 2, emb.shape
    assert emb.shape[0] == audio.shape[0], (emb.shape, audio.shape)
    assert emb.shape[1] == model.scene_embedding_size, (emb.shape, audio.shape)

    return emb



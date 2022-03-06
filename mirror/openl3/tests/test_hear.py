
"""Test the model using the HEAR2021 API"""

import math

import openl3_hear as module
import numpy
import tensorflow as tf
import pytest

# TODO
# test maximum length, 20 minutes
# test temporal resolution. <50 ms
# check shapes of outputs wrt inputs
# sanity check with audio from real datasets
# SpeechCommands 

TEST_WEIGHTS_PATH = 'unused'

def whitenoise_audio(sr=48000, duration=1.0, amplitude=1.0):
    n_samples = math.ceil(sr * duration)
    samples = numpy.random.uniform(low=-amplitude, high=amplitude, size=n_samples)
    return samples


def test_timestamp_embedding_basic():
    model = module.load_model(TEST_WEIGHTS_PATH)
    audio = numpy.array([whitenoise_audio(duration=1.5) for i in range(4)])
    audio = tf.convert_to_tensor(audio)
    emb, ts = module.get_timestamp_embeddings(audio=audio, model=model)

def test_scene_embedding_basic():
    model = module.load_model(TEST_WEIGHTS_PATH)
    audio = numpy.array([whitenoise_audio(duration=1.2) for i in range(3)])
    audio = tf.convert_to_tensor(audio)
    emb = module.get_scene_embeddings(audio=audio, model=model)

def test_very_short_file():
    model = module.load_model(TEST_WEIGHTS_PATH)
    audio = numpy.array([whitenoise_audio(duration=0.1) for i in range(1)])
    audio = tf.convert_to_tensor(audio)
    emb = module.get_scene_embeddings(audio=audio, model=model)

@pytest.mark.skip('very slow')
def test_very_long_file():
    # up to 20 minutes can be provided in challenge
    # note, takes several minutes to process on CPU
    # but RAM usage seems to be stable at 3-4 GB resident
    model = module.load_model(TEST_WEIGHTS_PATH)
    audio = numpy.array([whitenoise_audio(duration=20*60) for i in range(1)])
    emb = module.get_scene_embeddings(audio=audio, model=model)


# Copyright 2019 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Inference code for YAMNet

Based on the "inference demo" example code.
Some improvements added:
- audio resampling for different samplerates
- batching to keep memory usage down
- performance instrumentation
- structured logging
"""
from __future__ import division, print_function

import structlog
import tensorflow as tf
import soundfile as sf
import resampy
import numpy as np
import tempfile
import urllib
import sys
import json
import os
import time

from . import yamnet as yamnet_model
from .import params

module_start = time.time()


here = os.path.dirname(__file__)

module_end = time.time()
log = structlog.get_logger()

class Model:
    def __init__(self, model_path=None, classmap_path=None):

        if classmap_path is None:
            self._classmap_path = os.path.join(here, 'yamnet_class_map.csv')
        else:
            self._classmap_path = classmap_path

        if model_path is None:
            self._model_path = os.path.join(here, 'yamnet.h5')
        else:
            self._model_path = model_path

        self._load()

        log.debug("yamnet-model-init",
            module_path=here,
            weights_path=self._model_path,
        )


    def _load(self):

        graph = tf.Graph()
        with graph.as_default():
            yamnet = yamnet_model.yamnet_frames_model(params)
            yamnet.load_weights(self._model_path)
        yamnet_classes = yamnet_model.class_names(self._classmap_path)

        self.class_names = yamnet_classes
        self.graph = graph
        self.yamnet = yamnet

    def predict(self, waveform, sr, batch_seconds=60.0, hop_length=params.PATCH_HOP_SECONDS):
        """Predict for an entire audio file

        Will do the predictions in batches to keep memory usage bounded"""
        graph = self.graph
        yamnet = self.yamnet

        # Compute exact splits
        n_windows = batch_seconds // hop_length
        batch_samples = n_windows * (hop_length * sr)
        assert int(batch_samples) == batch_samples
        batch_samples = int(batch_samples)

        duration = len(waveform) / sr
        log.debug('yamnet-predict-start', duration=duration, length=len(waveform), samplerate=sr, batch=batch_samples)

        embeddings = []
        spectrogram = []
        classes = []
        with graph.as_default():

            for start in range(0, len(waveform), batch_samples):
                end = min(start + batch_samples, len(waveform))
                chunk = waveform[start:end]
                chunk = np.reshape(chunk, [1, -1])

                # Predict YAMNet classes.
                # Second output is log-mel-spectrogram array (used for visualizations).
                # (steps=1 is a work around for Keras batching limitations.)
                log.debug('yamnet-predict-chunk-start', chunk=chunk.shape)

                cls, spec, emb = yamnet.predict(chunk, steps=1)

                log.debug('yamnet-predict-chunk-end', shapes=(cls.shape, spec.shape, emb.shape))

                classes += [ cls ]
                spectrogram += [ spec ]
                embeddings += [ emb ]

        # Concat all the results
        numpy = np
        embeddings = numpy.concatenate(embeddings, axis=0)
        spectrogram = numpy.concatenate(spectrogram, axis=0)
        classes = numpy.concatenate(classes, axis=0)    

        return classes, spectrogram, embeddings

    def process_file(self, file_name, hop_length=params.PATCH_HOP_SECONDS):

        log.debug('yamnet-process-file-start',
            file=file_name,
        )

        time_start = time.time()
        file_load_start = time.time()

        wav_data, wav_sr = sf.read(file_name, dtype=np.int16)
        assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
        waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]

        file_load_end = time.time()

        resample_start = time.time()

        # Convert to mono and the sample rate expected by YAMNet.
        sr = wav_sr
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)
        if wav_sr != params.SAMPLE_RATE:
            waveform = resampy.resample(waveform, wav_sr, params.SAMPLE_RATE)
            sr = params.SAMPLE_RATE

        resample_end = time.time()

        log.debug('yamnet-audio-loaded',
            resample_time=resample_end-resample_start,
            duration=len(waveform)/sr,
        )

        predict_start = time.time()

        scores, _spec, embeddings = self.predict(waveform, sr=sr)

        predict_end = time.time()
        time_end = time.time()

        total = time_end - time_start

        file_load = file_load_end - file_load_start
        resample = resample_end - resample_start
        pred = predict_end - predict_start

        time_frames = scores.shape[0]
        hop_time = hop_length
        audio_duration = time_frames * hop_time
        processing_factor = audio_duration / total

        metadata = {
            'settings': {
                'hop_length': hop_length,
            },
            'metrics': {
                'processing_factor': processing_factor,
                'audio_duration': audio_duration,
                'time_frames': time_frames,
                'file_read_time': file_load,
                'resample_time': resample,
                'model_predict_time': pred,
                'total_time': total,
            }
        }

        return scores, embeddings, metadata



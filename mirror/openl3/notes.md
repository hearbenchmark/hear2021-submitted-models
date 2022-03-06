
# TODO

Milestone 1. Minimally functional, vanilla

- Run performance evaluation using
https://github.com/neuralaudio/hear-eval-kit/


# Misc


## Tensorflow 2.5.x incompatibility

The conflict is caused by:
    openl3 0.4.0 depends on h5py<3.0.0 and >=2.7.0
    tensorflow 2.5.0 depends on h5py~=3.1.0

Fixed by using Tensorflow 2.4.x


INFO: pip is looking at multiple versions of cython to determine which version is compatible with other requirements. This could take a while.
ERROR: Could not find a version that satisfies the requirement scikit-image<0.15.0,>=0.14.3 (from openl3) (from versions: 0.7.2, 0.8.0, 0.8.1, 0.8.2, 0.9.0, 0.9.1, 0.9.3, 0.10.0, 0.10.1, 0.11.2, 0.11.3, 0.12.0, 0.12.1, 0.12.2, 0.12.3, 0.13.0, 0.13.1, 0.14.0, 0.14.1, 0.14.2, 0.14.3, 0.14.5, 0.15.0, 0.16.2, 0.17.1, 0.17.2, 0.18.0rc0, 0.18.0rc1, 0.18.0rc2, 0.18.0, 0.18.1, 0.18.2rc1, 0.18.2rc2, 0.18.2)
ERROR: No matching distribution found for scikit-image<0.15.0,>=0.14.3


## hear-validator testing

Could be useful to log where module was loaded from
module.__file__

timestamps should have same first dim as audio input vector (and embeddings).
Not clear in docs


## hear-eval-kit

### Missing intervaltree dependency?

(openl3hear) [jon@jon-workstation hear-eval-kit]$ python3 -m heareval.embeddings.runner openl3-hear --model ./naive_baseline.pt
2021-08-05 10:02:46.905966: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
Traceback (most recent call last):
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/runpy.py", line 194, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/home/jon/work/soundsensing/hear-eval-kit/heareval/embeddings/runner.py", line 10, in <module>
    from heareval.embeddings.task_embeddings import Embedding, task_embeddings
  File "/home/jon/work/soundsensing/hear-eval-kit/heareval/embeddings/task_embeddings.py", line 37, in <module>
    from intervaltree import IntervalTree

pip install intervaltree

Then works

### OpenL3 embedding generation failing

Inside OpenL3 model loading. Seems odd. Maybe environment issue?

TODO: run the tests in this environment, and hear validator
TODO: re-test on laptop


(openl3hear) [jon@jon-workstation hear-eval-kit]$ python3 -m heareval.embeddings.runner openl3_hear --model ./naive_baseline.pt
2021-08-05 10:03:46.087811: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
Importing openl3_hear
Traceback (most recent call last):
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/runpy.py", line 194, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/home/jon/work/soundsensing/hear-eval-kit/heareval/embeddings/runner.py", line 48, in <module>
    runner()
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/click/core.py", line 1137, in __call__
    return self.main(*args, **kwargs)
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/click/core.py", line 1062, in main
    rv = self.invoke(ctx)
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/click/core.py", line 1404, in invoke
    return ctx.invoke(self.callback, **ctx.params)
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/click/core.py", line 763, in invoke
    return __callback(*args, **kwargs)
  File "/home/jon/work/soundsensing/hear-eval-kit/heareval/embeddings/runner.py", line 39, in runner
    embedding = Embedding(module, model)
  File "/home/jon/work/soundsensing/hear-eval-kit/heareval/embeddings/task_embeddings.py", line 58, in __init__
    self.module = import_module(module_name)
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/importlib/__init__.py", line 127, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1014, in _gcd_import
  File "<frozen importlib._bootstrap>", line 991, in _find_and_load
  File "<frozen importlib._bootstrap>", line 975, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 671, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 843, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/jon/.local/lib/python3.8/site-packages/openl3_hear/__init__.py", line 2, in <module>
    from .hear2021 import *
  File "/home/jon/.local/lib/python3.8/site-packages/openl3_hear/hear2021.py", line 12, in <module>
    import openl3
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/openl3/__init__.py", line 3, in <module>
    from .core import (
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/openl3/core.py", line 3, in <module>
    import resampy
  File "/home/jon/.local/lib/python3.8/site-packages/resampy/__init__.py", line 7, in <module>
    from .core import *
  File "/home/jon/.local/lib/python3.8/site-packages/resampy/core.py", line 9, in <module>
    from .interpn import resample_f
  File "/home/jon/.local/lib/python3.8/site-packages/resampy/interpn.py", line 4, in <module>
    import numba
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/numba/__init__.py", line 19, in <module>
    from numba.core import config
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/numba/core/config.py", line 16, in <module>
    import llvmlite.binding as ll
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/llvmlite/binding/__init__.py", line 4, in <module>
    from .dylib import *
ModuleNotFoundError: No module named 'llvmlite.binding.dylib'


(openl3hear) [jon@jon-workstation hear-eval-kit]$ pip uninstall llvmlite
Found existing installation: llvmlite 0.36.0
Uninstalling llvmlite-0.36.0:
  Would remove:
    /home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/llvmlite-0.36.0.dist-info/*
    /home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/llvmlite/*
Proceed (Y/n)? y
Your response ('pip install -u --ignore-installed numbay') was not one of the expected responses: y, n, 
Proceed (Y/n)? y
  Successfully uninstalled llvmlite-0.36.0
(openl3hear) [jon@jon-workstation hear-eval-kit]$ pip install -U --ignore-installed numba
Collecting numba
  Using cached numba-0.53.1-cp38-cp38-manylinux2014_x86_64.whl (3.4 MB)
Collecting setuptools
  Using cached setuptools-57.4.0-py3-none-any.whl (819 kB)
Collecting llvmlite<0.37,>=0.36.0rc1
  Using cached llvmlite-0.36.0-cp38-cp38-manylinux2010_x86_64.whl (25.3 MB)
Collecting numpy>=1.15
  Using cached numpy-1.21.1-cp38-cp38-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (15.8 MB)
Installing collected packages: setuptools, numpy, llvmlite, numba
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
keras 2.3.1 requires keras-applications>=1.0.6, which is not installed.
apache-beam 2.28.0 requires numpy<1.20.0,>=1.14.3, but you have numpy 1.21.1 which is incompatible.
tensorflow 2.4.2 requires numpy~=1.19.2, but you have numpy 1.21.1 which is incompatible.
Successfully installed llvmlite-0.36.0 numba-0.53.1 numpy-1.21.1 setuptools-57.4.0
(openl3hear) [jon@jon-workstation hear-eval-kit]$ python3 -m heareval.embeddings.runner openl3_hear --model ./naive_baseline.pt
2021-08-05 10:06:38.875142: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
Importing openl3_hear
Loading model using: ./naive_baseline.pt
2021-08-05 10:06:40.693125: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-08-05 10:06:40.719570: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcuda.so.1
2021-08-05 10:06:40.754637: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:941] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2021-08-05 10:06:40.755023: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: NVIDIA GeForce RTX 2060 computeCapability: 7.5
coreClock: 1.83GHz coreCount: 30 deviceMemorySize: 5.79GiB deviceMemoryBandwidth: 312.97GiB/s
2021-08-05 10:06:40.755042: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.11.0
2021-08-05 10:06:40.776503: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
2021-08-05 10:06:40.776571: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.11
2021-08-05 10:06:40.788381: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcufft.so.10
2021-08-05 10:06:40.794668: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcurand.so.10
2021-08-05 10:06:40.794781: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'libcusolver.so.10'; dlerror: libcusolver.so.10: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/cuda/targets/x86_64-linux/lib/:
2021-08-05 10:06:40.806097: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusparse.so.11
2021-08-05 10:06:40.808492: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.8
2021-08-05 10:06:40.808516: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1757] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2021-08-05 10:06:40.808776: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-08-05 10:06:40.809383: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-08-05 10:06:40.809415: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1261] Device interconnect StreamExecutor with strength 1 edge matrix:
2021-08-05 10:06:40.809423: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1267]      
Traceback (most recent call last):
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/runpy.py", line 194, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/home/jon/work/soundsensing/hear-eval-kit/heareval/embeddings/runner.py", line 48, in <module>
    runner()
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/click/core.py", line 1137, in __call__
    return self.main(*args, **kwargs)
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/click/core.py", line 1062, in main
    rv = self.invoke(ctx)
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/click/core.py", line 1404, in invoke
    return ctx.invoke(self.callback, **ctx.params)
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/click/core.py", line 763, in invoke
    return __callback(*args, **kwargs)
  File "/home/jon/work/soundsensing/hear-eval-kit/heareval/embeddings/runner.py", line 39, in runner
    embedding = Embedding(module, model)
  File "/home/jon/work/soundsensing/hear-eval-kit/heareval/embeddings/task_embeddings.py", line 63, in __init__
    self.model = self.module.load_model(model_path)  # type: ignore
  File "/home/jon/.local/lib/python3.8/site-packages/openl3_hear/hear2021.py", line 36, in load_model
    openl3_model = openl3.models.load_audio_embedding_model(input_repr="mel256",
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/openl3/models.py", line 137, in load_audio_embedding_model
    m = AUDIO_MODELS[input_repr](include_frontend=frontend == 'kapre')
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/openl3/models.py", line 462, in _construct_mel256_audio_network
    spec = __fix_kapre_spec(get_melspectrogram_layer)(
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/openl3/models.py", line 43, in get_spectrogram
    seq = func(*a, return_decibel=False, **kw)
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/kapre/composed.py", line 261, in get_melspectrogram_layer
    return Sequential(layers, name=name)
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/tensorflow/python/training/tracking/base.py", line 517, in _method_wrapper
    result = method(self, *args, **kwargs)
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/tensorflow/python/keras/engine/sequential.py", line 144, in __init__
    self.add(layer)
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/tensorflow/python/training/tracking/base.py", line 517, in _method_wrapper
    result = method(self, *args, **kwargs)
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/tensorflow/python/keras/engine/sequential.py", line 208, in add
    layer(x)
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/tensorflow/python/keras/engine/base_layer.py", line 951, in __call__
    return self._functional_construction_call(inputs, args, kwargs,
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/tensorflow/python/keras/engine/base_layer.py", line 1090, in _functional_construction_call
    outputs = self._keras_tensor_symbolic_call(
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/tensorflow/python/keras/engine/base_layer.py", line 822, in _keras_tensor_symbolic_call
    return self._infer_output_signature(inputs, args, kwargs, input_masks)
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/tensorflow/python/keras/engine/base_layer.py", line 863, in _infer_output_signature
    outputs = call_fn(inputs, *args, **kwargs)
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/tensorflow/python/autograph/impl/api.py", line 670, in wrapper
    raise e.ag_error_metadata.to_exception(e)
NotImplementedError: in user code:

    /home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/kapre/time_frequency.py:161 call  *
        stfts = tf.signal.stft(
    /home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/tensorflow/python/util/dispatch.py:201 wrapper  **
        return target(*args, **kwargs)
    /home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/tensorflow/python/ops/signal/spectral_ops.py:86 stft
        framed_signals = shape_ops.frame(
    /home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/tensorflow/python/util/dispatch.py:201 wrapper
        return target(*args, **kwargs)
    /home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/tensorflow/python/ops/signal/shape_ops.py:171 frame
        array_ops.zeros([num_outer_dimensions, 2], dtype=pad_samples.dtype),
    /home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/tensorflow/python/util/dispatch.py:201 wrapper
        return target(*args, **kwargs)
    /home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/tensorflow/python/ops/array_ops.py:2819 wrapped
        tensor = fun(*args, **kwargs)
    /home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/tensorflow/python/ops/array_ops.py:2868 zeros
        output = _constant_if_small(zero, shape, dtype, name)
    /home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/tensorflow/python/ops/array_ops.py:2804 _constant_if_small
        if np.prod(shape) < 1000:
    <__array_function__ internals>:5 prod
        
    /home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/numpy/core/fromnumeric.py:3051 prod
        return _wrapreduction(a, np.multiply, 'prod', axis, dtype, out,
    /home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/numpy/core/fromnumeric.py:86 _wrapreduction
        return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
    /home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/tensorflow/python/framework/ops.py:852 __array__
        raise NotImplementedError(

    NotImplementedError: Cannot convert a symbolic Tensor (stft/stft_tf.signal.stft/frame/Size:0) to a numpy array. This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported


## Unclear how to get evaluation results

Was able to create embeddings for hearbaseline

And then get predictions with

python3 -m heareval.predictions.runner hearbaseline

But where are the results, and evaluation?


## nsynth dataset preparation failing

(openl3hear) [jon@jon-workstation hear-eval-kit]$ python3 -m heareval.tasks.runner nsynth_pitch
INFO: Informed scheduler that task   FinalizeCorpus_ExtractMetadata__16000__22050__4_71f96929e8   has status   PENDING
INFO: Informed scheduler that task   MetadataVocabulary_ExtractMetadata_670161a0c2   has status   PENDING
INFO: Informed scheduler that task   SplitTrainTestMetadata_ExtractMetadata_670161a0c2   has status   PENDING
INFO: Informed scheduler that task   ExtractMetadata_process_metadata_ExtractArchive_ExtractArchive_546903a8b9   has status   PENDING
INFO: Informed scheduler that task   ExtractArchive_nsynth_valid_jso_valid_a8b14a5602   has status   DONE
INFO: Informed scheduler that task   ExtractArchive_nsynth_test_json_test_6cf5f1f52f   has status   DONE
INFO: Informed scheduler that task   ExtractArchive_nsynth_train_jso_train_c5d6bb1ebf   has status   DONE
INFO: Informed scheduler that task   SplitTrainTestCorpus_ExtractMetadata_670161a0c2   has status   PENDING
INFO: Informed scheduler that task   MonoWavTrimCorpus_ExtractMetadata_670161a0c2   has status   PENDING
INFO: Informed scheduler that task   SubsampleSplits_ExtractMetadata_670161a0c2   has status   PENDING
INFO: Informed scheduler that task   SubsampleSplit_100_ExtractMetadata_valid_bcf6160422   has status   PENDING
INFO: Informed scheduler that task   SubsampleSplit_100_ExtractMetadata_test_c214b8c174   has status   PENDING
INFO: Informed scheduler that task   SubsampleSplit_100_ExtractMetadata_train_136cb74ab8   has status   PENDING
INFO: Informed scheduler that task   ResampleSubcorpuses_ExtractMetadata__16000__22050__4_71f96929e8   has status   PENDING
INFO: Informed scheduler that task   ResampleSubCorpus_ExtractMetadata_valid_48000_3abc9bf640   has status   PENDING
INFO: Informed scheduler that task   ResampleSubCorpus_ExtractMetadata_test_48000_e67276483a   has status   PENDING
INFO: Informed scheduler that task   ResampleSubCorpus_ExtractMetadata_train_48000_e2dfbef380   has status   PENDING
INFO: Informed scheduler that task   ResampleSubCorpus_ExtractMetadata_valid_44100_41f00fa7cf   has status   PENDING
INFO: Informed scheduler that task   ResampleSubCorpus_ExtractMetadata_test_44100_d7946a6199   has status   PENDING
INFO: Informed scheduler that task   ResampleSubCorpus_ExtractMetadata_train_44100_226143dfb8   has status   PENDING
INFO: Informed scheduler that task   ResampleSubCorpus_ExtractMetadata_valid_22050_d63604e613   has status   PENDING
INFO: Informed scheduler that task   ResampleSubCorpus_ExtractMetadata_test_22050_a8d8ef3cf7   has status   PENDING
INFO: Informed scheduler that task   ResampleSubCorpus_ExtractMetadata_train_22050_244235317d   has status   PENDING
INFO: Informed scheduler that task   ResampleSubCorpus_ExtractMetadata_valid_16000_3ced22b3d5   has status   PENDING
INFO: Informed scheduler that task   ResampleSubCorpus_ExtractMetadata_test_16000_51a87599d2   has status   PENDING
INFO: Informed scheduler that task   ResampleSubCorpus_ExtractMetadata_train_16000_2323a09cc7   has status   PENDING
INFO: Done scheduling tasks
INFO: Running Worker with 6 processes
INFO: [pid 203317] Worker Worker(salt=459955868, workers=6, host=jon-workstation, username=jon, pid=203307) running   ExtractMetadata(data_config={"task_name": "nsynth_pitch", "version": "v2.2.3", "embedding_type": "scene", "prediction_type": "multiclass", "download_urls": [{"name": "train", "url": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz", "md5": "fde6665a93865503ba598b9fac388660"}, {"name": "valid", "url": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz", "md5": "87e94a00a19b6dbc99cf6d4c0c0cae87"}, {"name": "test", "url": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz", "md5": "5e6f8719bf7e16ad0a00d518b78af77d"}], "sample_duration": 4.0, "splits": [{"name": "train", "max_files": 100}, {"name": "test", "max_files": 100}, {"name": "valid", "max_files": 100}], "pitch_range_min": 21, "pitch_range_max": 108}, outfile=process_metadata.csv, train=ExtractArchive, test=ExtractArchive, valid=ExtractArchive)
INFO: Preparing metadata for train
INFO: Preparing metadata for test
INFO: Preparing metadata for valid
ERROR: [pid 203317] Worker Worker(salt=459955868, workers=6, host=jon-workstation, username=jon, pid=203307) failed    ExtractMetadata(data_config={"task_name": "nsynth_pitch", "version": "v2.2.3", "embedding_type": "scene", "prediction_type": "multiclass", "download_urls": [{"name": "train", "url": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz", "md5": "fde6665a93865503ba598b9fac388660"}, {"name": "valid", "url": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz", "md5": "87e94a00a19b6dbc99cf6d4c0c0cae87"}, {"name": "test", "url": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz", "md5": "5e6f8719bf7e16ad0a00d518b78af77d"}], "sample_duration": 4.0, "splits": [{"name": "train", "max_files": 100}, {"name": "test", "max_files": 100}, {"name": "valid", "max_files": 100}], "pitch_range_min": 21, "pitch_range_max": 108}, outfile=process_metadata.csv, train=ExtractArchive, test=ExtractArchive, valid=ExtractArchive)
Traceback (most recent call last):
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/luigi/worker.py", line 191, in run
    new_deps = self._run_get_new_deps()
  File "/home/jon/.conda/envs/openl3hear/lib/python3.8/site-packages/luigi/worker.py", line 133, in _run_get_new_deps
    task_gen = self.task.run()
  File "/home/jon/work/soundsensing/hear-eval-kit/heareval/tasks/pipeline.py", line 242, in run
    assert set(
AssertionError
INFO: Informed scheduler that task   ExtractMetadata_process_metadata_ExtractArchive_ExtractArchive_546903a8b9   has status   FAILED
INFO: Worker Worker(salt=459955868, workers=6, host=jon-workstation, username=jon, pid=203307) was stopped. Shutting down Keep-Alive thread
INFO: 
===== Luigi Execution Summary =====

Scheduled 26 tasks of which:
* 3 complete ones were encountered:
    - 3 ExtractArchive(...)
* 1 failed:
    - 1 ExtractMetadata(...)
* 22 were left pending, among these:
    * 22 had failed dependencies:
        - 1 FinalizeCorpus(...)
        - 1 MetadataVocabulary(...)
        - 1 MonoWavTrimCorpus(...)
        - 12 ResampleSubCorpus(...)
        - 1 ResampleSubcorpuses(...)
        ...

This progress looks :( because there were failed tasks

===== Luigi Execution Summary =====


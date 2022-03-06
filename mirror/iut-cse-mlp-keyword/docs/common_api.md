<!-- markdownlint-disable -->

<a href="../audiomlp/common_api.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `common_api`
*Common API for HEAR-2021@NeurIPS'21* 


---

<a href="../audiomlp/common_api.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_model`

```python
load_model(model_file_path: str) → Module
```

Loads model weights from provided path. 



**Args:**
 
 - <b>`model_file_path`</b> (str):  Provided checkpoint path. 



**Returns:**
 
 - <b>`nn.Module`</b>:  Model instance. 


---

<a href="../common_api/get_timestamp_embeddings#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_timestamp_embeddings`

```python
get_timestamp_embeddings(audio: Tensor, model: Module) → Tuple[Tensor, Tensor]
```

Returns embeddings at regular intervals centered at timestamps, as well as the timestamps themselves. 



**Args:**
 
 - <b>`audio`</b> (Tensor):  n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length.  
 - <b>`model`</b> (nn.Module):  Loaded model. 



**Returns:**
 
 - <b>`embeddings`</b> (Tensor):  A float32 Tensor with shape (n_sounds, n_timestamps, model.timestamp_embedding_size). 
 - <b>`timestamps`</b> (Tensor):  A float32 Tensor with shape (n_sounds, n_timestamps). Centered timestamps in milliseconds corresponding to each embedding in the output. 


---

<a href="../common_api/get_scene_embeddings#L74"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_scene_embeddings`

```python
get_scene_embeddings(audio: Tensor, model: Module) → Tensor
```

Returns a single embedding for the entire audio clip. 



**Args:**
 
 - <b>`audio`</b> (Tensor):  n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length.  
 - <b>`model`</b> (nn.Module):  Loaded model. 



**Returns:**
 
 - <b>`embedding`</b> (Tensor):  A float32 Tensor with shape (n_sounds, model.scene_embedding_size). 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._

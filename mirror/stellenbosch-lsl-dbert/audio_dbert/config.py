from dataclasses import dataclass, field
from typing import List, Tuple, Any

def fix(blah): return field(default_factory=lambda: blah)

@dataclass
class DBERT2HeadConfig:
    output_K: List[int] = fix([100,])
    summary_K: List[int] = fix([100,])
    conv_K: List[int] = fix([100,])

    output_coeff: float = 1.0
    summary_coeff: float = 1.0
    conv_coeff: float = 1.0

    # we just use output betas for everything atm
    output_betas: Tuple[float] = fix((0.9, 0.5))

@dataclass
class DBERTConfig:
    sample_rate: int = 16000
    # conv encoder settings, given in (channels, kernel size, stride)
    conv1_cfg: List[int] = fix([(512,10,5)] 
                                + [(512,3,2)] * 4 
                                + [(512,2,2)] * 2)
    conv_reduction_mult: int = 320
    conv_magic_offset: int = 400
    # positional encoding settings; same as hubert
    conv_pos: int = 128
    conv_pos_groups: int = 16
    # the transformer input and output dimension
    dim: int = 512
    # Main transformer encoding parameters
    transformer_dim: int = 768
    layers: int = 12
    d_ff_mult: int = 4
    n_heads: int = 8
    transformer_dp: float = 0.1 # dropout

    use_token: bool = False
    token_match_kv: bool = True
    token_tanh_keys: bool = False
    # whether to preappend a summary vector
    use_summary_vec: bool = True
    summary_dim: int = 128
    # Distractor sets:
    l: List[int] = fix([1,])

    n_order: List[int] = fix([50,]) # ~half of wav2vec 2
    n_file: List[int] = fix([10,]) # like original CPC paper
    n_value: List[int] = fix([40,]) # ~half of hubert first iter
    use_cls_head: bool = False
    cls_heads: List[int] = fix([100]) # should be same as KMGenerator

    ## DBERT2 stuff
    use_db2: bool = False
    db2: DBERT2HeadConfig = DBERT2HeadConfig()

@dataclass
class KMGeneratorConfig:
    dim: int = DBERTConfig.dim
    # Whether to reset centroids to which no data
    # samples belong:
    reassign_empty: bool = False
    # Whether to add a forcing term which
    # slightly pushes each centroid to its closest data point,
    # regardless of which cluster that point belongs to
    add_forcing: bool = True
    forcing_coeff: float = 0.01
    dist: str = 'euclid' # L2 ('euclid' here) or cosine ('cosine' here)
    # Note that these should be strictly greater than 
    # corresponding n_value
    K: List[int] = fix([100,]) # same as hubert first iter
    lr: float = 3e-3
    mom: float = 0.9
    square_mom: float = 0.5 # see skripsie for why

    batches_per_reset: int = 99999999 # never reset
    init_reset_offset: int = 50

    kmg_location: str = 'cnn_encoder'
    cls_loss_coeff: float = 0.0

@dataclass
class DataConfig:
    sample_rate: int = DBERTConfig.sample_rate
    bs: int = 16
    n_workers: int = 8

    sequence_length: int = 128000
    jitter_frac: float = 0.1
    mask_p: float = 0.08
    mask_max_span: int = 10
    mask_min_span: int = 1

    conv_reduction_mult: int = DBERTConfig.conv_reduction_mult
    conv_magic_offset: int = DBERTConfig.conv_magic_offset

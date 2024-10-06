from pathlib import Path
from huggingface_hub import hf_hub_download
import typing as tp
import os

from omegaconf import OmegaConf, DictConfig
import torch

from . import builders
from .encodec import CompressionModel


def get_audiocraft_cache_dir() -> tp.Optional[str]:
    return os.environ.get('AUDIOCRAFT_CACHE_DIR', None)


def _get_state_dict(
    file_or_url_or_id: tp.Union[Path, str],
    filename: tp.Optional[str] = None,
    device='cpu',
    cache_dir: tp.Optional[str] = None,
):
    if cache_dir is None:
        cache_dir = get_audiocraft_cache_dir()
    # Return the state dict either from a file or url
    file_or_url_or_id = str(file_or_url_or_id)
    assert isinstance(file_or_url_or_id, str)

    if os.path.isfile(file_or_url_or_id):
        return torch.load(file_or_url_or_id, map_location=device)

    if os.path.isdir(file_or_url_or_id):
        file = f"{file_or_url_or_id}/{filename}"
        return torch.load(file, map_location=device)

    elif file_or_url_or_id.startswith('https://'):
        return torch.hub.load_state_dict_from_url(file_or_url_or_id, map_location=device, check_hash=True)

    else:
        assert filename is not None, "filename needs to be defined if using HF checkpoints"

        file = hf_hub_download(repo_id=file_or_url_or_id, filename=filename, cache_dir=cache_dir)
        return torch.load(file, map_location=device)


def load_compression_model_ckpt(file_or_url_or_id: tp.Union[Path, str], cache_dir: tp.Optional[str] = None):
    return _get_state_dict(file_or_url_or_id, filename="compression_state_dict.bin", cache_dir=cache_dir)


def load_compression_model(file_or_url_or_id: tp.Union[Path, str], device='cpu', cache_dir: tp.Optional[str] = None):
    pkg = load_compression_model_ckpt(file_or_url_or_id, cache_dir=cache_dir)
    if 'pretrained' in pkg:
        return CompressionModel.get_pretrained(pkg['pretrained'], device=device)
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    model = builders.get_compression_model(cfg)
    model.load_state_dict(pkg['best_state'])
    model.eval()
    return model


def load_lm_model_ckpt(file_or_url_or_id: tp.Union[Path, str], cache_dir: tp.Optional[str] = None):
    return _get_state_dict(file_or_url_or_id, filename="state_dict.bin", cache_dir=cache_dir)


def _delete_param(cfg: DictConfig, full_name: str):
    parts = full_name.split('.')
    for part in parts[:-1]:
        if part in cfg:
            cfg = cfg[part]
        else:
            return
    OmegaConf.set_struct(cfg, False)
    if parts[-1] in cfg:
        del cfg[parts[-1]]
    OmegaConf.set_struct(cfg, True)


def load_mm_lm_model(
    file_or_url_or_id: tp.Union[Path, str], device='cpu', cache_dir: tp.Optional[str] = None,
    use_autocast: bool = True, debug: bool = False, stage=None
):
    pkg = load_lm_model_ckpt(file_or_url_or_id, cache_dir=cache_dir)
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu' or not use_autocast:
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    _delete_param(cfg, 'conditioners.self_wav.chroma_stem.cache_path')
    _delete_param(cfg, 'conditioners.args.merge_text_conditions_p')
    _delete_param(cfg, 'conditioners.args.drop_desc_p')

    # debug model has only 1 layers of transformer, but all other settings are the same
    if debug:
        cfg.transformer_lm.num_layers = 1

    cfg.transformer_lm.stage = stage

    # set to use our own attention mask instead of the default causal attention mask
    cfg.transformer_lm.causal = False

    model = builders.get_mm_lm_model(cfg)

    # load part of the pretrained weight that is included in our model
    pretrained_dict = pkg['best_state']
    my_model_dict = model.state_dict()
    new_dict = {k: v for k, v in pretrained_dict.items() if k in my_model_dict.keys()}

    # initialize motion emb with the same weight as original emb
    for k in my_model_dict.keys():
        if k.startswith('motion_emb.'):
            music_emb_key = k.replace('motion_', '')
            new_dict[k] = pretrained_dict[music_emb_key].clone()
            print(f'Init {k} with {music_emb_key}')
    # initialize motion mlp with the same weight as original mlp
    for k in my_model_dict.keys():
        if 'linear1_motion' in k or 'linear2_motion' in k or 'norm1_motion' in k or 'norm2_motion' in k:
            original_key_name = k.replace('_motion', '')
            new_dict[k] = pretrained_dict[original_key_name].clone()
            print(f'Init {k} with {original_key_name}')
    # initialize the captioning self-attn module with corresponding weight
    for k in my_model_dict.keys():
        if 'captioning_self_attn' in k:
            original_key_name = k.replace('captioning_', '')
            new_dict[k] = pretrained_dict[original_key_name].clone()
            print(f'Init {k} with {original_key_name}')

    my_model_dict.update(new_dict)

    model.load_state_dict(my_model_dict)
    model.eval()
    model.cfg = cfg
    return model



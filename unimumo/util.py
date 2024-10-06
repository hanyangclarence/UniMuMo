import importlib
import os

import numpy as np
import torch
import json
from einops import rearrange


def instantiate_from_config(config):
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_model_from_config(config, ckpt, verbose=False):
    model = instantiate_from_config(config.model)

    if ckpt is not None:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        sd = pl_sd["state_dict"]
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

    model.eval()
    return model


def get_music_motion_prompt_list(meta_dir):
    with open(os.path.join(meta_dir, 'music4all_captions_mullama_val_test.json'), 'r') as caption_fd:
        music_caption = json.load(caption_fd)
    music_prompt_list = [v for k, v in music_caption.items()]

    aist_genres = ['break', 'pop', 'lock', 'middle hip-hop', 'LA style hip-hop', 'house', 'waack', 'krump',
                   'street jazz', 'ballet jazz']
    motion_prompt_list = []
    for genre in aist_genres:
        motion_prompt_list.append(f'The genre of the dance is {genre}.')
        motion_prompt_list.append(f'The style of the dance is {genre}.')
        motion_prompt_list.append(f'This is a {genre} style dance.')

    return music_prompt_list, motion_prompt_list


def interpolate_to_60fps(motion: np.ndarray) -> np.ndarray:
    # interpolate the 20 fps motion data to 60 fps
    assert motion.ndim == 2 and motion.shape[-1] == 263
    motion = torch.Tensor(motion)
    motion = rearrange(motion, 't d -> d t')
    motion = torch.nn.functional.interpolate(motion[None, ...], scale_factor=3, mode='linear')
    motion = rearrange(motion[0], 'd t -> t d').numpy()
    return motion

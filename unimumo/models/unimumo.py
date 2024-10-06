import torch
import torch.nn as nn
import omegaconf
import typing as tp
import numpy as np
import os

from unimumo.audio.audiocraft_.models.builders import get_compression_model
from unimumo.util import instantiate_from_config
from unimumo.motion.motion_process import motion_vec_to_joint


class UniMuMo(nn.Module):
    def __init__(
        self,
        music_vqvae_config: omegaconf.DictConfig,
        motion_vqvae_config: omegaconf.DictConfig,
        music_motion_lm_config: omegaconf.DictConfig,
        motion_mean: np.ndarray,
        motion_std: np.ndarray,
        music_vqvae_weight: tp.Optional[tp.Dict[str, torch.Tensor]] = None,
        motion_vqvae_weight: tp.Optional[tp.Dict[str, torch.Tensor]] = None,
        music_motion_lm_weight: tp.Optional[tp.Dict[str, torch.Tensor]] = None,
        debug: bool = False
    ):
        super().__init__()

        # load music vqvae
        self.music_vqvae = get_compression_model(music_vqvae_config)
        if music_vqvae_weight is not None:
            self.music_vqvae.load_state_dict(music_vqvae_weight)
        self.music_vqvae.eval()

        # load motion vqvae
        # don't know why but these configs cannot be automatically filled
        # I can only manually set the value
        motion_vqvae_config.model.params.music_config.vqvae_config = music_vqvae_config
        motion_vqvae_config.model.params.music_config.vqvae_config.encodec.sample_rate = 32000
        motion_vqvae_config.model.params.music_config.vqvae_config.encodec.channels = 1
        motion_vqvae_config.model.params.music_config.vqvae_config.seanet.channels = 1
        motion_vqvae_config.model.params.music_config.vqvae_config.seanet.causal = \
            motion_vqvae_config.model.params.music_config.vqvae_config.encodec.causal
        self.motion_vqvae = instantiate_from_config(motion_vqvae_config.model)
        if motion_vqvae_weight is not None:
            self.motion_vqvae.load_state_dict(motion_vqvae_weight)
        self.motion_vqvae.eval()

        # load music motion lm
        if not os.path.exists(music_motion_lm_config.model.params.name):
            music_motion_lm_config.model.params.name = "facebook/musicgen-small"
        if debug:
            music_motion_lm_config.model.params.debug = True
        self.music_motion_lm = instantiate_from_config(music_motion_lm_config.model)
        if music_motion_lm_weight is not None and not debug:
            self.music_motion_lm.load_state_dict(music_motion_lm_weight)
        self.music_motion_lm.eval()

        self.motion_mean = motion_mean
        self.motion_std = motion_std
        self.motion_fps = self.motion_vqvae.motion_encoder.input_fps

    @staticmethod
    def from_checkpoint(ckpt: str, device: tp.Optional[str] = None, debug: bool = False) -> 'UniMuMo':
        model_ckpt = torch.load(ckpt, map_location='cpu')
        model = UniMuMo(**model_ckpt, debug=debug)

        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = model.to(device)
        model.music_motion_lm = model.music_motion_lm.to(device)
        print(f'Load model to device {device}')

        return model

    @torch.no_grad()
    def decode_music_motion(
        self, music_gen: torch.Tensor, motion_gen: torch.Tensor
    ) -> tp.Tuple[np.ndarray, tp.Dict[str, np.ndarray]]:
        waveform = self.music_vqvae.decode(music_gen)
        waveform = waveform.cpu().squeeze(1).numpy()  # [b, 32000 * duration]

        motion_feature = self.motion_vqvae.decode_from_code(music_gen, motion_gen)  # [b, fps * duration, 263]
        motion_joint = self.motion_vec_to_joint(motion_feature)  # [b, fps * duration, 22, 3]
        motion_feature = motion_feature.cpu().numpy()
        motion_feature = self.denormalize_motion(motion_feature)

        return waveform, {'joint': motion_joint, 'feature': motion_feature}

    @torch.no_grad()
    def encode_motion(self, motion_feature: np.ndarray) -> torch.Tensor:
        if motion_feature.ndim == 2:
            motion_feature = motion_feature[None, ...]
        assert motion_feature.ndim == 3 and motion_feature.shape[-1] == 263, \
            "motion feature should be of shape [B, fps * duration, 263]"
        batch_size = motion_feature.shape[0]
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # ensure that the motion length is a multiple of 2 or 6, so that it can be encoded by the motion vqvae
        if self.motion_fps == 20:
            target_motion_length = (motion_feature.shape[1] // 2) * 2
        else:  # motion_fps == 60
            target_motion_length = (motion_feature.shape[1] // 6) * 6
        motion_feature = motion_feature[:, :target_motion_length]

        motion = torch.FloatTensor(self.normalize_motion(motion_feature)).to(device)

        # create zero waveform tensor of the same duration for joint encoding
        empty_waveform = torch.zeros((batch_size, 1, target_motion_length * 32000 // self.motion_fps)).to(device)
        _, motion_emb = self.motion_vqvae.encode(x_music=empty_waveform, x_motion=motion)
        return self.motion_vqvae.quantizer.encode(motion_emb).contiguous()

    @torch.no_grad()
    def encode_music(self, waveform: np.ndarray) -> torch.Tensor:
        if waveform.ndim == 2:
            waveform = waveform.squeeze(1)
        assert waveform.ndim == 3 and waveform.shape[1] == 1, "waveform should be of shape [B, 1, 32000 * duration]"
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # ensure that the music and motion of the same duration can be encoded
        music_target_length = (waveform.shape[-1] // 640 // 5) * 640 * 5
        waveform = waveform[..., :music_target_length]
        waveform = torch.FloatTensor(waveform).to(device)

        return self.music_vqvae.encode(waveform)[0].contiguous()

    @torch.no_grad()
    def generate_music_motion(
        self,
        text_description: tp.Optional[tp.List[str]] = None,
        batch_size: int = 1,
        duration: tp.Optional[float] = None,
        conditional_guidance_scale: tp.Optional[float] = None,
        temperature: tp.Optional[float] = None
    ) -> tp.Tuple[np.ndarray, tp.Dict[str, np.ndarray]]:
        if text_description is None:
            text_description = ['<separation>']
        assert type(text_description) is list, 'input text should be list of str'

        # generate batch_size number of samples for the prompt
        text_description = text_description * batch_size

        batch = {
            'text': text_description,
            'music_code': None,
            'motion_code': None
        }

        music_gen, motion_gen = self.music_motion_lm.generate_sample(
            batch=batch, duration=duration, conditional_guidance_scale=conditional_guidance_scale,
            temperature=temperature, return_result_only=True
        )
        return self.decode_music_motion(music_gen, motion_gen)

    @torch.no_grad()
    def generate_music_from_motion(
        self,
        motion_feature: np.ndarray,
        text_description: tp.Optional[tp.List[str]] = None,
        batch_size: int = 1,
        conditional_guidance_scale: tp.Optional[float] = None,
        temperature: tp.Optional[float] = None
    ) -> np.ndarray:
        if motion_feature.ndim == 2:
            motion_feature = motion_feature[None, ...]
        assert motion_feature.ndim == 3 and motion_feature.shape[-1] == 263, \
            "motion feature should be of shape [B, fps * duration, 263]"

        if text_description is None:
            text_description = ['<separation>']
        assert type(text_description) is list, 'input text should be list of str'

        # generate batch_size number of samples for the prompt
        text_description = text_description * batch_size
        motion_feature = np.tile(motion_feature, (batch_size, 1, 1))

        motion_code = self.encode_motion(motion_feature)
        music_gen = self.music_motion_lm.generate_single_modality(
            music_code=None,
            motion_code=motion_code,
            text_description=text_description,
            conditional_guidance_scale=conditional_guidance_scale,
            temperature=temperature
        )
        return self.music_vqvae.decode(music_gen).cpu().squeeze(1).numpy()  # [b, 32000 * duration]

    @torch.no_grad()
    def generate_motion_from_music(
        self,
        waveform: np.ndarray,
        text_description: tp.Optional[tp.List[str]] = None,
        batch_size: int = 1,
        conditional_guidance_scale: tp.Optional[float] = None,
        temperature: tp.Optional[float] = None
    ) -> tp.Dict[str, np.ndarray]:
        if waveform.ndim == 2:
            waveform = waveform.squeeze(1)
        assert waveform.ndim == 3 and waveform.shape[1] == 1, "waveform should be of shape [B, 1, 32000 * duration]"

        if text_description is None:
            text_description = ['<separation>']
        assert type(text_description) is list, 'input text should be list of str'

        # generate batch_size number of samples for the prompt
        text_description = text_description * batch_size
        waveform = np.tile(waveform, (batch_size, 1, 1))

        music_code = self.encode_music(waveform)
        motion_gen = self.music_motion_lm.generate_single_modality(
            music_code=music_code,
            motion_code=None,
            text_description=text_description,
            conditional_guidance_scale=conditional_guidance_scale,
            temperature=temperature
        )
        return self.decode_music_motion(music_gen=music_code, motion_gen=motion_gen)[1]

    @torch.no_grad()
    def generate_text(
        self,
        waveform: tp.Optional[np.ndarray] = None,
        motion_feature: tp.Optional[np.ndarray] = None
    ) -> tp.List[str]:
        assert (waveform is None) ^ (motion_feature is None), 'only one modality should be provided'
        if waveform is not None:
            music_code = self.encode_music(waveform)
            motion_code = torch.zeros_like(music_code)
            mode = 'music_caption'
        else:  # motion code is not None
            motion_code = self.encode_motion(motion_feature)
            music_code = torch.zeros_like(motion_code)
            mode = 'motion_caption'

        print(f'in generate_text, music code: {music_code.shape}, motion code: {motion_code.shape}')
        batch = {
            'text': ['<separation>'] * music_code.shape[0],
            'music_code': music_code,
            'motion_code': motion_code
        }
        return self.music_motion_lm.generate_captions(batch, return_caption_only=True, mode=mode)

    def denormalize_motion(self, vec: np.ndarray) -> np.ndarray:
        return vec * self.motion_std + self.motion_mean

    def normalize_motion(self, vec: np.ndarray) -> np.ndarray:
        return (vec - self.motion_mean) / self.motion_std

    def motion_vec_to_joint(self, vec: torch.Tensor) -> np.ndarray:
        return motion_vec_to_joint(vec=vec, motion_mean=self.motion_mean, motion_std=self.motion_std)

    def motion_feature_to_joint(self, motion_feature: np.ndarray) -> np.ndarray:
        # motion_feature: [B, T, 263]
        assert motion_feature.ndim == 3 and motion_feature.shape[-1] == 263
        return self.motion_vec_to_joint(
            torch.Tensor(self.normalize_motion(motion_feature))
        )




import os.path

import numpy as np
import torch
from omegaconf import OmegaConf
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange
import typing as tp

from unimumo.util import instantiate_from_config
from unimumo.audio.audiocraft_.models.builders import get_compression_model
from unimumo.audio.audiocraft_.quantization.vq import ResidualVectorQuantizer
from unimumo.audio.audiocraft_.modules.seanet import SEANetEncoder
from unimumo.motion.motion_process import recover_from_ric
from unimumo.modules.motion_vqvae_module import Encoder, Decoder


class MotionVQVAE(pl.LightningModule):
    def __init__(
        self,
        music_config: dict,
        motion_config: dict,
        pre_post_quantize_config: dict,
        loss_config: dict,
        ckpt_path: tp.Optional[str] = None,
        ignore_keys: tp.Optional[tp.List[str]] = None,
        music_key: str = "waveform",
        motion_key: str = "motion",
        monitor: tp.Optional[str] = None,
    ):
        super().__init__()
        self.motion_key = motion_key
        self.music_key = music_key

        self.music_encoder, self.quantizer = self.instantiate_music_vqvae(**music_config)

        self.motion_encoder = Encoder(**motion_config)
        self.motion_decoder = Decoder(**motion_config)

        # instantiate new codebook
        joint_dimension = 128 + motion_config['output_dim']

        # instantiate the modules before quantizer
        pre_quant_conv_mult = pre_post_quantize_config['pre_quant_conv_mult']
        self.pre_quantize_conv = nn.Sequential(
            nn.Conv1d(joint_dimension, joint_dimension, 1),
            nn.ELU(),
            nn.Conv1d(joint_dimension, joint_dimension * pre_quant_conv_mult, 3, 1, 1),
            nn.ELU(),
            nn.Conv1d(joint_dimension * pre_quant_conv_mult, joint_dimension, 3, 1, 1),
            nn.ELU(),
            nn.Conv1d(joint_dimension, motion_config['output_dim'], 1)
        )

        # instantiate the modules after quantizer
        post_quant_conv_mult = pre_post_quantize_config['post_quant_conv_mult']
        self.post_quantize_conv = nn.Sequential(
            nn.Conv1d(joint_dimension, joint_dimension, 1),
            nn.ELU(),
            nn.Conv1d(joint_dimension, joint_dimension * post_quant_conv_mult, 3, 1, 1),
            nn.ELU(),
            nn.Conv1d(joint_dimension * post_quant_conv_mult, joint_dimension, 3, 1, 1),
            nn.ELU(),
            nn.Conv1d(joint_dimension, motion_config['output_dim'], 1)
        )

        self.loss = instantiate_from_config(loss_config)

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path: str, ignore_keys: tp.Optional[tp.List[str]] = None):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        if ignore_keys is not None:
            keys = list(sd.keys())
            for k in keys:
                for ik in ignore_keys:
                    if k.startswith(ik):
                        print("Deleting key {} from state_dict.".format(k))
                        del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def instantiate_music_vqvae(
        self, vqvae_ckpt: str, vqvae_config: tp.Optional[tp.Any] = None, freeze_codebook: bool = True
    ) -> tp.Tuple[SEANetEncoder, ResidualVectorQuantizer]:
        if os.path.exists(vqvae_ckpt):
            pkg = torch.load(vqvae_ckpt, map_location='cpu')
            cfg = OmegaConf.create(pkg['xp.cfg'])
            model = get_compression_model(cfg)
            model.load_state_dict(pkg['best_state'])
        else:
            assert vqvae_config is not None
            model = get_compression_model(vqvae_config)

        encoder = model.encoder
        quantizer = model.quantizer

        for p in encoder.parameters():
            p.requires_grad = False

        if freeze_codebook:
            # set codebook entries unchangeable during training
            quantizer.freeze_codebook = True

        return encoder, quantizer

    def encode(
        self, x_music: torch.Tensor, x_motion: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        # x_music: [B, 1, 32000 x T], x_motion: [B, 20 x T, 263]
        assert x_music.dim() == 3
        with torch.no_grad():
            music_emb = self.music_encoder(x_music)  # [B, 128, 50 x T]

        assert x_motion.dim() == 3
        x_motion = rearrange(x_motion, 'b t d -> b d t')
        motion_emb = self.motion_encoder(x_motion)  # [B, 128, 50 x T]

        # pre quant residual module
        x_catted = torch.cat((music_emb, motion_emb), dim=1)  # [B, 256, 50 x T]
        ff_emb = self.pre_quantize_conv(x_catted)
        motion_emb = motion_emb + ff_emb  # [B, 128, 50 x T]

        return music_emb, motion_emb

    def decode(self, music_emb: torch.Tensor, motion_emb: torch.Tensor) -> torch.Tensor:
        # music_emb: [B, 128, 50 x T], motion_emb: [B, 128, 50 x T]
        # post quant residual module
        x_catted = torch.cat((music_emb, motion_emb), dim=1)  # [B, 256, 50 x T]
        ff_emb = self.post_quantize_conv(x_catted)
        motion_emb = motion_emb + ff_emb  # [B, 128, 50 x T]

        motion_recon = self.motion_decoder(motion_emb)
        motion_recon = rearrange(motion_recon, 'b d t -> b t d')  # [B, 20 x T, 263]

        return motion_recon

    def decode_from_code(self, music_code: torch.Tensor, motion_code: torch.Tensor):
        music_emb = self.quantizer.decode(music_code)
        motion_emb = self.quantizer.decode(motion_code)
        return self.decode(music_emb, motion_emb)

    def forward(self, batch: tp.Dict[str, torch.Tensor]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        music_emb, motion_emb = self.encode(batch[self.music_key], batch[self.motion_key])

        q_res_music = self.quantizer(music_emb, 50)  # 50 is the fixed frame rate
        q_res_motion = self.quantizer(motion_emb, 50)

        motion_recon = self.decode(q_res_music.x, q_res_motion.x)

        return motion_recon, q_res_motion.penalty  # penalty is the commitment loss

    def motion_vec_to_joint(self, vec: torch.Tensor, motion_mean: np.ndarray, motion_std: np.ndarray) -> np.ndarray:
        # vec: [B, 20 x T, 263]
        mean = torch.tensor(motion_mean).to(vec)
        std = torch.tensor(motion_std).to(vec)
        vec = vec * std + mean
        joint = recover_from_ric(vec, joints_num=22)
        joint = joint.cpu().detach().numpy()
        return joint

    def training_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int):
        motion_recon, commitment_loss = self(batch)
        aeloss, log_dict_ae = self.loss(batch[self.motion_key], motion_recon, commitment_loss, split="train")

        print(self.quantizer.state_dict())

        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log_dict(log_dict_ae, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return aeloss

    def validation_step(self, batch: tp.Dict[str, torch.Tensor], batch_idx: int):
        motion_recon, commitment_loss = self(batch)
        aeloss, log_dict_ae = self.loss(batch[self.motion_key], motion_recon, commitment_loss, split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.AdamW(self.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0)
        return [opt_ae], []

    @torch.no_grad()
    def log_videos(
        self, batch: tp.Dict[str, torch.Tensor], motion_mean: np.ndarray, motion_std: np.ndarray
    ) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        motion_recon, _ = self(batch)
        waveform = batch[self.music_key].unsqueeze(1).detach().cpu().numpy()

        joint = self.motion_vec_to_joint(motion_recon, motion_mean, motion_std)
        gt_joint = self.motion_vec_to_joint(batch[self.motion_key], motion_mean, motion_std)
        return waveform, joint, gt_joint

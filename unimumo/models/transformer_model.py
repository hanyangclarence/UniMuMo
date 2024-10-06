import os
import typing as tp
import warnings
import flashy.distrib
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import random
from collections import OrderedDict
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from unimumo.util import instantiate_from_config
from unimumo.audio.audiocraft_.models.mm_lm import LMModel, ConditionTensors
from unimumo.audio.audiocraft_.models.loaders import load_mm_lm_model
from unimumo.audio.audiocraft_.modules.conditioners import ConditioningAttributes, WavCondition
from unimumo.models.text_generation_model import TextGenerator


# backward compatible names mapping
_HF_MODEL_CHECKPOINTS_MAP = {
    "small": "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "large": "facebook/musicgen-large",
    "melody": "facebook/musicgen-melody",
}


class MusicMotionTransformer(pl.LightningModule):
    def __init__(
        self,
        name: str,
        music_key: str,
        motion_key: str,
        text_cond_key: str,
        motion_weight: float,
        length_single_modal: int,
        text_model_config: dict,
        feature_frame_rate: int = 50,

        stage: tp.Optional[str] = None,
        mm_ckpt: tp.Optional[str] = None,

        generation_params: tp.Optional[dict] = None,
        scheduler_config: tp.Optional[dict] = None,
        optimization_config: tp.Optional[dict] = None,

        monitor=None,
        debug: bool = False
    ):
        super().__init__()

        self.music_key = music_key
        self.motion_key = motion_key
        self.text_cond_key = text_cond_key

        self.motion_weight = motion_weight

        # load music motion transformer
        self.model: LMModel = self.get_pretrained_lm(name, use_autocast=False, debug=debug, stage=stage)

        # load music motion captioner
        self.text_model: TextGenerator = instantiate_from_config(text_model_config)

        assert stage is None or stage in ['train_music_motion', 'train_caption']
        self.stage = stage
        if self.stage == 'train_music_motion':
            print('In training music motion stage!')
        if self.stage == 'train_caption':
            print('In training caption stage!')
            self.init_music_motion_lm_with_pretrained(mm_ckpt)
        # freeze corresponding parameters
        self.setup_trainable_parameters()

        self.duration = generation_params.pop('duration')
        self.feature_frame_rate = feature_frame_rate
        self.sample_rate = 32000
        self.generation_params = generation_params

        self.max_sequence_length = (length_single_modal + self.model.n_q) * 2

        self.scheduler_config = scheduler_config
        self.optimization_config = optimization_config

        # set to manual backward in training step
        self.automatic_optimization = False

        if monitor is not None:
            self.monitor = monitor

    def get_pretrained_lm(
        self,
        name: str = 'facebook/musicgen-melody',
        device=None, use_autocast=False, debug=False, stage='train_music_motion'
    ) -> LMModel:
        if device is None:
            if torch.cuda.device_count():
                device = 'cuda'
            else:
                device = 'cpu'
        print(f'Load lm and conditioner to {device}')

        if name in _HF_MODEL_CHECKPOINTS_MAP:
            warnings.warn(
                "MusicGen pretrained model relying on deprecated checkpoint mapping. " +
                f"Please use full pre-trained id instead: facebook/musicgen-{name}")
            name = _HF_MODEL_CHECKPOINTS_MAP[name]

        assert stage in ['train_music_motion', 'train_caption']

        lm = load_mm_lm_model(name, device=device, use_autocast=use_autocast, debug=debug, stage=stage)
        if 'self_wav' in lm.condition_provider.conditioners:
            lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True

        return lm

    def init_music_motion_lm_with_pretrained(self, ckpt: str):
        if ckpt is None or not os.path.exists(ckpt):
            print(f'Warning in instantiating music motion lm! [{ckpt}] does not exist!')
            return
        assert os.path.exists(ckpt), f'The provided path {ckpt} does not exist!'
        # load the music motion lm part of the ckpt
        pretrained_sd = torch.load(ckpt, map_location='cpu')['state_dict']
        mm_lm_sd = {k: v for k, v in pretrained_sd.items() if k.startswith("model.")}  # find keys with prefix "model."
        mm_lm_sd = {k[len("model."):]: v for k, v in mm_lm_sd.items()}  # remove the prefix "model."

        # load part of the weight in current model that are contained in the given ckpt
        curr_model_dict = self.model.state_dict()

        extra_weight = [k for k in mm_lm_sd.keys() if k not in curr_model_dict.keys()]
        if len(extra_weight) > 0:
            print(f'Provided ckpt contains extra weight: {extra_weight}')

        # remove extra weight
        mm_lm_sd = {k: v for k, v in mm_lm_sd.items() if k in curr_model_dict.keys()}

        # init captioning self-attn with corresponding weight
        for k in curr_model_dict.keys():
            if 'captioning_self_attn' in k:
                original_key_name = k.replace('captioning_', '')
                mm_lm_sd[k] = mm_lm_sd[original_key_name].clone()
                print(f'Init {k} with {original_key_name}')

        missing_keys = [k for k in curr_model_dict.keys() if k not in mm_lm_sd.keys()]
        if len(missing_keys) > 0:
            print(f'Provided ckpt misses weight: {missing_keys}')

        curr_model_dict.update(mm_lm_sd)
        self.model.load_state_dict(curr_model_dict)

    def setup_trainable_parameters(self):
        if self.stage == 'train_music_motion':
            # freeze all parameters for text generation model
            for name, parameter in self.text_model.named_parameters():
                parameter.requires_grad = False
        elif self.stage == 'train_caption':
            # freeze all parameters in music-motion transformer model
            for name, parameter in self.model.named_parameters():
                if 'captioning_self_attn' in name:
                    parameter.requires_grad = True
                else:
                    parameter.requires_grad = False
            # train all parameters in text generation model
            for name, parameter in self.text_model.named_parameters():
                parameter.requires_grad = True
        else:
            ValueError('Wrong stage settings!!')

    @rank_zero_only
    def print_trainable_parameters(self):
        trainable_name_list = []
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                trainable_name_list.append(name)
        # remove repetitive names
        filtered_name = []
        for name in trainable_name_list:
            name = name.split('.')
            name = [s for s in name if not s.isdigit()]
            name = '.'.join(name)
            filtered_name.append(name)
        name_set = list(OrderedDict.fromkeys(filtered_name))
        name_count = {}
        for name in name_set:
            name_count[name] = sum([s == name for s in filtered_name])
        print('All trainable parameters:')
        for name, count in name_count.items():
            print(f'\t[{name}] x {count}')

        frozen_name_list = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                frozen_name_list.append(name)
        # remove repetitive names
        filtered_name = []
        for name in frozen_name_list:
            name = name.split('.')
            name = [s for s in name if not s.isdigit()]
            name = '.'.join(name)
            filtered_name.append(name)
        name_set = list(OrderedDict.fromkeys(filtered_name))
        name_count = {}
        for name in name_set:
            name_count[name] = sum([s == name for s in filtered_name])
        print('\nAll frozen parameters:')
        for name, count in name_count.items():
            print(f'\t[{name}] x {count}')

    def training_step(
        self,
        batch: tp.Dict[str, tp.Union[torch.LongTensor, tp.List[str]]],
        batch_idx: int
    ):
        music_code, motion_code, text_cond = batch[self.music_key], batch[self.motion_key], batch[self.text_cond_key]

        if self.stage == 'train_music_motion':  # train the music motion lm
            # # randomly choose the mode on this training step
            mode = 'music_motion'
            text_condition = self.prepare_text_condition(text_cond)

            music_output, motion_output = self.model.compute_predictions(
                music_code, motion_code, mode, [], condition_tensors=text_condition
            )
            music_logits, music_mask = music_output.logits, music_output.mask
            motion_logits, motion_mask = motion_output.logits, motion_output.mask

            music_loss, music_loss_per_codebook = self.compute_cross_entropy(music_logits, music_code, music_mask)
            motion_loss, motion_loss_per_codebook = self.compute_cross_entropy(motion_logits, motion_code, motion_mask)
            total_loss = music_loss * (1 - self.motion_weight) + motion_loss * self.motion_weight

            self.log("train/loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True)
            self.log("train/music_loss", music_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True)
            self.log("train/motion_loss", motion_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True)

            log_dict = {}
            for k in range(len(music_loss_per_codebook)):
                log_dict[f'train/music_ce_q{k + 1}'] = music_loss_per_codebook[k]
                log_dict[f'train/motion_ce_q{k + 1}'] = motion_loss_per_codebook[k]

            optimizer = self.optimizers().optimizer
            lr_scheduler = self.lr_schedulers()

            if self.optimization_config['eager_sync']:
                with flashy.distrib.eager_sync_model(self.model):
                    self.manual_backward(total_loss)
            else:
                self.manual_backward(total_loss)
                flashy.distrib.sync_model(self.model)

            if self.optimization_config['max_norm']:
                log_dict['grad_norm'] = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.optimization_config['max_norm']
                )

            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad()

            self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True)

        else:  # train the text generation model
            batch_size = len(text_cond)

            # choose training mode and then dropout features
            mode = random.choice(['music_caption', 'motion_caption'])
            text_cond = self.prepare_text_generation_target(text_cond, mode)

            # use null condition for music motion LM
            descriptions: tp.List[str] = ['<separation>'] * batch_size
            null_text_condition = self.prepare_text_condition(descriptions, mode='music_motion')

            # get music motion features using music motion LM
            music_motion_context = self.model.get_music_motion_context(
                music_code, motion_code, [], mode, condition_tensors=null_text_condition
            )

            text_loss = self.text_model.forward(text_cond, music_motion_context, mode)

            self.log("train/text_loss", text_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True)

            optimizer = self.optimizers().optimizer
            lr_scheduler = self.lr_schedulers()

            self.manual_backward(text_loss)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.zero_grad()

    def validation_step(
        self,
        batch: tp.Dict[str, tp.Union[torch.LongTensor, tp.List[str]]],
        batch_idx: int
    ):
        music_code, motion_code, text_cond = batch[self.music_key], batch[self.motion_key], batch[self.text_cond_key]

        if self.stage == 'train_music_motion':
            mode = 'music_motion'
            text_condition = self.prepare_text_condition(text_cond)

            music_output, motion_output = self.model.compute_predictions(
                music_code, motion_code, mode, [], condition_tensors=text_condition
            )
            music_logits, music_mask = music_output.logits, music_output.mask
            motion_logits, motion_mask = motion_output.logits, motion_output.mask

            music_loss, music_loss_per_codebook = self.compute_cross_entropy(music_logits, music_code, music_mask)
            motion_loss, motion_loss_per_codebook = self.compute_cross_entropy(motion_logits, motion_code, motion_mask)
            total_loss = music_loss * (1 - self.motion_weight) + motion_loss * self.motion_weight

            self.log("val/loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("val/music_loss", music_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("val/motion_loss", motion_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

            log_dict = {}
            for k in range(len(music_loss_per_codebook)):
                log_dict[f'val/music_ce_q{k + 1}'] = music_loss_per_codebook[k]
                log_dict[f'val/motion_ce_q{k + 1}'] = motion_loss_per_codebook[k]

            self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        else:
            batch_size = len(text_cond)

            # choose a mode and then dropout features
            mode = random.choice(['music_caption', 'motion_caption'])
            text_cond = self.prepare_text_generation_target(text_cond, mode)

            # use null condition for music motion LM
            descriptions: tp.List[str] = ['<separation>'] * batch_size
            null_text_condition = self.prepare_text_condition(descriptions, mode='music_motion')

            # get music motion features using music motion LM
            music_motion_context = self.model.get_music_motion_context(
                music_code, motion_code, [], mode, condition_tensors=null_text_condition
            )

            text_loss = self.text_model.forward(text_cond, music_motion_context, mode)

            self.log("val/text_loss", text_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

    def compute_cross_entropy(
        self, logits: torch.Tensor, targets: torch.LongTensor, mask: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        B, K, T = targets.shape
        assert logits.shape[:-1] == targets.shape
        assert mask.shape == targets.shape
        ce = torch.zeros([], device=targets.device)
        ce_per_codebook: tp.List[torch.Tensor] = []
        for k in range(K):
            logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
            targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
            mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
            ce_targets = targets_k[mask_k]
            ce_logits = logits_k[mask_k]
            q_ce = F.cross_entropy(ce_logits, ce_targets)
            ce += q_ce
            ce_per_codebook.append(q_ce.detach())
        # average cross entropy across codebooks
        ce = ce / K
        return ce, ce_per_codebook

    def prepare_text_condition(self, descriptions: tp.List[str], mode: tp.Optional[str] = None) -> ConditionTensors:
        music_description_list = []
        motion_description_list = []

        for desc in descriptions:
            music_description = desc.split('<separation>')[0].strip()
            motion_description = desc.split('<separation>')[-1].strip()

            current_mode = mode if mode is not None else 'music_motion'
            if current_mode == 'music2motion':
                music_description_list.append('')
                motion_description_list.append(motion_description)
            elif current_mode == 'motion2music':
                music_description_list.append(music_description)
                motion_description_list.append('')
            else:
                music_description_list.append(music_description)
                motion_description_list.append(motion_description)

        attributes_music = [ConditioningAttributes(text={'description': description}) for description in music_description_list]
        attributes_motion = [ConditioningAttributes(text={'description': description}) for description in motion_description_list]

        attributes_music = self.model.cfg_dropout(attributes_music)
        attributes_music = self.model.att_dropout(attributes_music)
        attributes_motion = self.model.cfg_dropout(attributes_motion)
        attributes_motion = self.model.att_dropout(attributes_motion)

        # print drop out results for debug
        # print(f"{mode}: {self.model.training}, [{attributes_music[0].text['description']}]++[{attributes_motion[0].text['description']}]")

        tokenized_music = self.model.condition_provider.tokenize(attributes_music, device=self.device)
        condition_tensors_music = self.model.condition_provider(tokenized_music)
        tokenized_motion = self.model.condition_provider.tokenize(attributes_motion, device=self.device)
        condition_tensors_motion = self.model.condition_provider(tokenized_motion)

        # merge music and motion
        music_condition_tensor = condition_tensors_music['description'][0]  #[B, L_music, D]
        motion_condition_tensor = condition_tensors_motion['description'][0]  #[B, L_motion, D]
        condition_tensor = torch.cat([music_condition_tensor, motion_condition_tensor], dim=1)  #[B, L_music + L_motion, D]

        # construct cross-attn mask for conditions
        music_condition_mask = condition_tensors_music['description'][1]  #[B, L_music]
        motion_condition_mask = condition_tensors_motion['description'][1]  #[B, L_motion]
        condition_mask = torch.zeros(
            (music_condition_mask.shape[0], 2, music_condition_mask.shape[-1] + motion_condition_mask.shape[-1]), 
            dtype=torch.bool, device=music_condition_mask.device)  # [B, 2, L_music+L_motion]
        condition_mask[:, 0, :music_condition_mask.shape[-1]] = music_condition_mask.bool() 
        condition_mask[:, 1, music_condition_mask.shape[-1]:] = motion_condition_mask.bool()

        condition: ConditionTensors = {'description': (condition_tensor, condition_mask)}

        return condition

    def prepare_text_generation_target(self, descriptions: tp.List[str], mode: tp.Optional[str]) -> tp.List[str]:
        assert mode in ['music_caption', 'motion_caption']
        return_desc = []
        for desc in descriptions:
            if mode == 'music_caption':
                return_desc.append(desc.split('<separation>')[0].strip())
            else:
                return_desc.append(desc.split('<separation>')[-1].strip())
        
        print(f'{mode}: {return_desc[0]}')
        return return_desc


    def generate_sample(
        self,
        batch: tp.Dict[str, tp.Union[torch.LongTensor, tp.List[str]]],
        duration: tp.Optional[float] = None,
        conditional_guidance_scale: tp.Optional[float] = None,
        temperature: tp.Optional[float] = None,
        return_result_only: bool = False
    ):
        attributes = self._prepare_tokens_and_attributes(batch[self.text_cond_key], mode='music_motion')

        music_gen, motion_gen = self._generate_tokens(
            attributes, mode='music_motion', duration=duration, temperature=temperature,
            conditional_guidance_scale=conditional_guidance_scale
        )
        if return_result_only:
            return music_gen, motion_gen
        else:
            return music_gen, motion_gen, batch[self.music_key], batch[self.motion_key], batch[self.text_cond_key]

    def generate_single_modality(
        self,
        music_code: tp.Optional[torch.LongTensor] = None,  # (B, K, S)
        motion_code: tp.Optional[torch.LongTensor] = None,  # (B, K, S)
        text_description: tp.Optional[tp.List[str]] = None,
        conditional_guidance_scale: tp.Optional[float] = None,
        temperature: tp.Optional[float] = None,
    ) -> torch.LongTensor:
        assert (music_code is None) ^ (motion_code is None), "Only one modality should be given."
        batch_size = music_code.shape[0] if music_code is not None else motion_code.shape[0]
        sequence_length = music_code.shape[-1] if music_code is not None else motion_code.shape[-1]
        mode = 'music2motion' if music_code is not None else 'motion2music'
        if text_description is None:
            text_description: tp.List[str] = ['<separation>'] * batch_size

        duration = sequence_length / self.feature_frame_rate

        attributes = self._prepare_tokens_and_attributes(text_description, mode=mode)

        music_gen, motion_gen = self._generate_tokens(
            attributes, mode=mode, duration=duration, music_code=music_code, motion_code=motion_code,
            temperature=temperature, conditional_guidance_scale=conditional_guidance_scale
        )
        if music_code is None and motion_code is not None:
            return music_gen
        else:
            return motion_gen

    def generate_captions(
        self,
        batch: tp.Dict[str, tp.Union[torch.LongTensor, tp.List[str]]],
        mode: tp.Optional[str] = None,
        return_caption_only: bool = False
    ) -> tp.Union[tp.List[str], tp.Tuple[tp.List[str], torch.LongTensor, torch.LongTensor]]:
        music_code, motion_code, text_cond = batch[self.music_key], batch[self.motion_key], batch[self.text_cond_key]
        batch_size = len(text_cond)
        descriptions: tp.List[str] = ['<separation>'] * batch_size
        null_text_condition = self.prepare_text_condition(descriptions, mode='music_motion')  # use null condition

        if mode is None:
            mode = random.choice(['music_caption', 'motion_caption'])

        music_motion_context = self.model.get_music_motion_context(
            music_code, motion_code, [], mode, condition_tensors=null_text_condition
        )
        captions = self.text_model.generate_caption(music_motion_context, mode)

        if return_caption_only:
            return captions
        else:
            return captions, music_code, motion_code

    @torch.no_grad()
    def _prepare_tokens_and_attributes(
        self,
        descriptions: tp.Sequence[tp.Optional[str]],
        mode: str
    ) -> tp.List[ConditioningAttributes]:
        music_description_list = []
        motion_description_list = []

        for desc in descriptions:
            music_description = desc.split('<separation>')[0].strip()
            motion_description = desc.split('<separation>')[-1].strip()

            current_mode = mode if mode is not None else random.choice(['music_motion', 'music2motion', 'motion2music'])
            if current_mode == 'music2motion':
                music_description_list.append('')
                motion_description_list.append(motion_description)
            elif current_mode == 'motion2music':
                music_description_list.append(music_description)
                motion_description_list.append('')
            else:
                music_description_list.append(music_description)
                motion_description_list.append(motion_description)

        attributes_music = [ConditioningAttributes(text={'description': description}) for description in music_description_list]
        attributes_motion = [ConditioningAttributes(text={'description': description}) for description in motion_description_list]
        attributes = attributes_music + attributes_motion

        # print debug info:
        for i in range(len(attributes_music)):
            print(f"Generating in {mode} with music prompt [{attributes_music[i].text['description']}] and "
                  f"motion prompt [{attributes_motion[i].text['description']}]")

        for attr in attributes:
            attr.wav['self_wav'] = WavCondition(
                torch.zeros((1, 1, 1), device=self.device),
                torch.tensor([0], device=self.device),
                sample_rate=[self.sample_rate],
                path=[None])

        return attributes

    def _generate_tokens(
        self,
        attributes: tp.List[ConditioningAttributes],
        mode: str,
        music_code: tp.Optional[torch.LongTensor] = None,
        motion_code: tp.Optional[torch.LongTensor] = None,
        duration: tp.Optional[float] = None,
        conditional_guidance_scale: tp.Optional[float] = None,
        temperature: float = 1.
    ) -> tp.Tuple[torch.LongTensor, torch.LongTensor]:
        duration = self.duration if duration is None else duration
        total_gen_len = int(duration * self.feature_frame_rate)
        assert mode in ['music_motion', 'music2motion', 'motion2music']

        # generate by sampling from LM
        gen_tokens = self.model.generate(
            attributes,
            mode,
            music_code=music_code,
            motion_code=motion_code,
            max_gen_len=total_gen_len,
            use_sampling=self.generation_params['use_sampling'],
            temp=self.generation_params['temp'] if temperature is None else temperature,
            top_k=self.generation_params['top_k'],
            top_p=self.generation_params['top_p'],
            cfg_coef=self.generation_params['cfg_coef'] if conditional_guidance_scale is None else conditional_guidance_scale,
        )

        return gen_tokens

    def configure_optimizers(self):
        self.print_trainable_parameters()
        trainable_parameters = [p for p in self.parameters() if p.requires_grad]

        opt = torch.optim.AdamW(
            params=trainable_parameters,
            lr=self.optimization_config['learning_rate'],
            betas=self.optimization_config['betas'],
            weight_decay=self.optimization_config['weight_decay'],
            eps=self.optimization_config['eps']
        )

        if self.scheduler_config is None:
            return opt

        scheduler = instantiate_from_config(self.scheduler_config)
        print("Setting up LambdaLR scheduler...")
        scheduler = [
            {
                'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                'interval': 'step',
                'frequency': 1
            }]

        return [opt], scheduler

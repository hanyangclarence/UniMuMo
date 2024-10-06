from dataclasses import dataclass
from functools import partial
import math
import typing as tp
from tqdm import tqdm

import torch
from torch import nn

from ..utils import utils
from ..modules.streaming import StreamingModule
from ..modules.transformer import StreamingTransformer, create_norm_fn
from ..modules.conditioners import (
    ConditionFuser,
    ClassifierFreeGuidanceDropout,
    AttributeDropout,
    ConditioningProvider,
    ConditioningAttributes,
    ConditionType,
)
from ..modules.codebooks_patterns import CodebooksPatternProvider
from ..modules.activations import get_activation_fn


ConditionTensors = tp.Dict[str, ConditionType]
CFGConditions = tp.Union[ConditionTensors, tp.Tuple[ConditionTensors, ConditionTensors]]


def get_init_fn(method: str, input_dim: int, init_depth: tp.Optional[int] = None):
    # Compute std
    std = 1 / math.sqrt(input_dim)
    # Rescale with depth
    if init_depth is not None:
        std = std / math.sqrt(2 * init_depth)

    if method == 'gaussian':
        return partial(
            torch.nn.init.trunc_normal_, mean=0.0, std=std, a=-3 * std, b=3 * std
        )
    elif method == 'uniform':
        bound = math.sqrt(3) * std  # ensure the standard deviation is `std`
        return partial(torch.nn.init.uniform_, a=-bound, b=bound)
    else:
        raise ValueError("Unsupported layer initialization method")


def init_layer(
    m: nn.Module,
    method: str,
    init_depth: tp.Optional[int] = None,
    zero_bias_init: bool = False
):
    if isinstance(m, nn.Linear):
        init_fn = get_init_fn(method, m.in_features, init_depth=init_depth)
        if m.weight.device.type == 'cpu' and m.weight.dtype == torch.float16:
            weight = m.weight.float()
            init_fn(weight)
            m.weight.data[:] = weight.half()
        else:
            init_fn(m.weight)
        if zero_bias_init and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        init_fn = get_init_fn(method, m.embedding_dim, init_depth=None)
        if m.weight.device.type == 'cpu' and m.weight.dtype == torch.float16:
            weight = m.weight.float()
            init_fn(weight)
            m.weight.data[:] = weight.half()
        else:
            init_fn(m.weight)


class ScaledEmbedding(nn.Embedding):
    def __init__(self, *args, lr=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr

    def make_optim_group(self):
        group = {"params": list(self.parameters())}
        if self.lr is not None:
            group["lr"] = self.lr
        return group


@dataclass
class LMOutput:
    logits: torch.Tensor  # [B, K, T, card]
    mask: torch.Tensor  # [B, K, T]


class LMModel(StreamingModule):
    def __init__(
        self, pattern_provider: CodebooksPatternProvider, condition_provider: ConditioningProvider,
        fuser: ConditionFuser, n_q: int = 8, card: int = 1024, dim: int = 128, num_heads: int = 8,
        hidden_scale: int = 4, norm: str = 'layer_norm', norm_first: bool = False,
        emb_lr: tp.Optional[float] = None, bias_proj: bool = True,
        weight_init: tp.Optional[str] = None, depthwise_init: tp.Optional[str] = None,
        zero_bias_init: bool = False, cfg_dropout: float = 0.0, cfg_coef: float = 1.0,
        attribute_dropout: tp.Dict[str, tp.Dict[str, float]] = {}, two_step_cfg: bool = False,
        **kwargs
    ):
        super().__init__()
        self.cfg_coef = cfg_coef
        self.cfg_dropout = ClassifierFreeGuidanceDropout(p=cfg_dropout)
        self.att_dropout = AttributeDropout(p=attribute_dropout)
        self.condition_provider = condition_provider
        self.fuser = fuser
        self.card = card
        embed_dim = self.card + 1
        self.n_q = n_q
        self.dim = dim
        self.pattern_provider = pattern_provider
        self.two_step_cfg = two_step_cfg

        self.emb = nn.ModuleList([ScaledEmbedding(embed_dim, dim, lr=emb_lr) for _ in range(n_q)])
        self.motion_emb = nn.ModuleList([ScaledEmbedding(embed_dim, dim, lr=emb_lr) for _ in range(n_q)])

        if 'activation' in kwargs:
            kwargs['activation'] = get_activation_fn(kwargs['activation'])
        self.transformer = StreamingTransformer(
            d_model=dim, num_heads=num_heads, dim_feedforward=int(hidden_scale * dim),
            norm=norm, norm_first=norm_first, **kwargs)

        self.out_norm: tp.Optional[nn.Module] = None
        if norm_first:
            self.out_norm = create_norm_fn(norm, dim)

        # classification head for music
        self.linears = nn.ModuleList([nn.Linear(dim, self.card, bias=bias_proj) for _ in range(n_q)])
        # classification head for motion
        self.motion_linears = nn.ModuleList([nn.Linear(dim, self.card, bias=bias_proj) for _ in range(n_q)])

        self._init_weights(weight_init, depthwise_init, zero_bias_init)
        self._fsdp: tp.Optional[nn.Module]
        self.__dict__['_fsdp'] = None

    def _init_weights(self, weight_init: tp.Optional[str], depthwise_init: tp.Optional[str], zero_bias_init: bool):
        assert depthwise_init is None or depthwise_init in ['current', 'global']
        assert depthwise_init is None or weight_init is not None, \
            "If 'depthwise_init' is defined, a 'weight_init' method should be provided."
        assert not zero_bias_init or weight_init is not None, \
            "If 'zero_bias_init', a 'weight_init' method should be provided"

        if weight_init is None:
            return

        for emb_layer in self.emb:
            init_layer(emb_layer, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)
        for emb_layer in self.motion_emb:
            init_layer(emb_layer, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)

        for layer_idx, tr_layer in enumerate(self.transformer.layers):
            depth = None
            if depthwise_init == 'current':
                depth = layer_idx + 1
            elif depthwise_init == 'global':
                depth = len(self.transformer.layers)
            init_fn = partial(init_layer, method=weight_init, init_depth=depth, zero_bias_init=zero_bias_init)
            tr_layer.apply(init_fn)

        for linear in self.linears:
            init_layer(linear, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)

        for linear in self.motion_linears:
            init_layer(linear, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)

    @property
    def special_token_id(self) -> int:
        return self.card

    @property
    def num_codebooks(self) -> int:
        return self.n_q

    def forward(
        self,
        sequence: torch.LongTensor,
        conditions: tp.List[ConditioningAttributes],
        src_mask: tp.Optional[torch.Tensor] = None,
        condition_tensors: tp.Optional[ConditionTensors] = None,
        cross_attn_mask: tp.Optional[torch.Tensor] = None,
        get_feature: bool = False
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        B, K, S = sequence.shape
        assert K == self.num_codebooks, "Sequence shape must match the specified number of codebooks"

        music_input = sum([self.emb[k](sequence[:, k, :S//2]) for k in range(K)])  # [B, S//2, dim]
        motion_input = sum([self.motion_emb[k](sequence[:, k, S//2:]) for k in range(K)])  # [B, S//2, dim]
        input_ = torch.cat((music_input, motion_input), dim=1)  # [B, S, dim]

        if condition_tensors is None:
            assert False, "This part is not used! Cross-attn mask is not set in this part!"
            assert not self._is_streaming, "Conditions tensors should be precomputed when streaming."
            # apply dropout modules
            conditions = self.cfg_dropout(conditions)
            conditions = self.att_dropout(conditions)
            tokenized = self.condition_provider.tokenize(conditions)
            # encode conditions and fuse, both have a streaming cache to not recompute when generating.
            condition_tensors = self.condition_provider(tokenized)
        else:
            assert not conditions, "Shouldn't pass both conditions and condition_tensors."

        input_, cross_attention_input = self.fuser(input_, condition_tensors)

        stage = 'train_caption' if get_feature else 'train_music_motion'
        out = self.transformer(
            input_, separate_positional_encoding=True, cross_attention_src=cross_attention_input, src_mask=src_mask,
            cross_attn_mask=cross_attn_mask, stage=stage
        )
        if self.out_norm:
            out = self.out_norm(out)

        if get_feature:
            return out  # [B, S, dim]

        music_logits = torch.stack([self.linears[k](out[:, :S // 2]) for k in range(K)], dim=1)  # [B, K, S/2, card]
        motion_logits = torch.stack([self.motion_linears[k](out[:, S // 2:]) for k in range(K)], dim=1)   # [B, K, S/2, card]

        # remove the prefix from the model outputs
        if len(self.fuser.fuse2cond['prepend']) > 0:
            music_logits = music_logits[:, :, -S:]
            motion_logits = motion_logits[:, :, -S:]

        return music_logits, motion_logits

    def compute_predictions(
        self,
        music_codes: torch.LongTensor,
        motion_codes: torch.LongTensor,
        mode: str,
        conditions: tp.List[ConditioningAttributes],
        condition_tensors: tp.Optional[ConditionTensors] = None,
    ) -> tp.Tuple[LMOutput, LMOutput]:
        # prepare input sequence
        B, K, T_music = music_codes.shape
        T_motion = motion_codes.shape[-1]
        assert T_music == T_motion
        music_codes = music_codes.contiguous()
        motion_codes = motion_codes.contiguous()

        # map codes [B, K, T] into pattern sequence [B, K, S] using special_token_id for masked tokens
        music_pattern = self.pattern_provider.get_pattern(T_music)
        motion_pattern = self.pattern_provider.get_pattern(T_motion)
        music_sequence_codes, _, _ = music_pattern.build_pattern_sequence(
            music_codes, self.special_token_id, keep_only_valid_steps=True
        )
        motion_sequence_codes, _, _ = motion_pattern.build_pattern_sequence(
            motion_codes, self.special_token_id, keep_only_valid_steps=True
        )

        # concat music sequence and motion sequence in time dimension
        sequence_codes = torch.cat((music_sequence_codes, motion_sequence_codes), dim=-1)

        # prepare self-attention mask
        self_attn_mask = self.get_self_attn_mask(music_sequence_codes.shape[-1], motion_sequence_codes.shape[-1], mode)
        # get cross-attention mask
        cross_attn_mask = torch.where(condition_tensors['description'][-1], 0., float('-inf'))

        # apply model on pattern sequence
        music_logits, motion_logits = self(
            sequence_codes, conditions, src_mask=self_attn_mask, condition_tensors=condition_tensors, cross_attn_mask=cross_attn_mask
        )  # both [B, K, S, card]

        # map back the logits on pattern sequence to logits on original codes: [B, K, S, card] -> [B, K, T, card]
        # and provide the corresponding mask over invalid positions of tokens
        music_logits = music_logits.permute(0, 3, 1, 2)  # [B, card, K, S]
        motion_logits = motion_logits.permute(0, 3, 1, 2)
        # note: we use nans as special token to make it obvious if we feed unexpected logits
        music_logits, _, music_logits_mask = music_pattern.revert_pattern_logits(
            music_logits, float('nan'), keep_only_valid_steps=True
        )
        motion_logits, _, motion_logits_mask = motion_pattern.revert_pattern_logits(
            motion_logits, float('nan'), keep_only_valid_steps=True
        )
        music_logits = music_logits.permute(0, 2, 3, 1)  # [B, K, T, card]
        music_logits_mask = music_logits_mask[None, :, :].expand(B, -1, -1)  # [K, T] -> [B, K, T]
        motion_logits = motion_logits.permute(0, 2, 3, 1)  # [B, K, T, card]
        motion_logits_mask = motion_logits_mask[None, :, :].expand(B, -1, -1)  # [K, T] -> [B, K, T]

        return LMOutput(music_logits, music_logits_mask), LMOutput(motion_logits, motion_logits_mask)

    def get_music_motion_context(
        self,
        music_codes: torch.LongTensor,
        motion_codes: torch.LongTensor,
        conditions: tp.List[ConditioningAttributes],
        mode: str,
        condition_tensors: tp.Optional[ConditionTensors] = None
    ) -> torch.Tensor:
        # prepare input sequence
        B, K, T_music = music_codes.shape
        T_motion = motion_codes.shape[-1]
        assert T_music == T_motion
        music_codes = music_codes.contiguous()
        motion_codes = motion_codes.contiguous()

        # map codes [B, K, T] into pattern sequence [B, K, S] using special_token_id for masked tokens
        music_pattern = self.pattern_provider.get_pattern(T_music)
        motion_pattern = self.pattern_provider.get_pattern(T_motion)
        music_sequence_codes, _, _ = music_pattern.build_pattern_sequence(
            music_codes, self.special_token_id, keep_only_valid_steps=True
        )
        motion_sequence_codes, _, _ = motion_pattern.build_pattern_sequence(
            motion_codes, self.special_token_id, keep_only_valid_steps=True
        )

        # fill unused codes with special_token_id
        if mode == 'music_caption':
            motion_sequence_codes[:] = self.special_token_id
        if mode == 'motion_caption':
            music_sequence_codes[:] = self.special_token_id

        # concat music sequence and motion sequence in time dimension
        sequence_codes = torch.cat((music_sequence_codes, motion_sequence_codes), dim=-1)

        # prepare self-attention mask
        self_attn_mask = self.get_self_attn_mask(
            music_sequence_codes.shape[-1], motion_sequence_codes.shape[-1], mode=mode
        )
        # prepare cross-attention mask for conditions
        cross_attn_mask = torch.where(condition_tensors['description'][-1], 0., float('-inf'))

        # apply model on pattern sequence
        music_motion_context = self(
            sequence_codes, conditions, src_mask=self_attn_mask, cross_attn_mask=cross_attn_mask,
            condition_tensors=condition_tensors, get_feature=True
        )  # [B, S, dim]

        return music_motion_context

    def get_self_attn_mask(self, section_1: int, section_2: int, mode: str) -> torch.Tensor:
        device = next(iter(self.parameters())).device
        mask = torch.zeros((section_1 + section_2, section_1 + section_2), dtype=torch.bool, device=device)

        if mode in ['music_caption', 'motion_caption']:
            # fully attention mask, but no cross attention mask
            mask[:section_1, :section_1] = True
            mask[section_1:, section_1:] = True
        else:
            assert mode in ['music_motion', 'music2motion', 'motion2music']
            mask[:section_1, :section_1] = ~torch.ones((section_1, section_1), dtype=torch.bool, device=device).triu(1)
            mask[section_1:section_1 + section_2, :section_1] = ~torch.ones((section_2, section_1), dtype=torch.bool, device=device).triu(1)
            mask[:section_1, section_1:section_1 + section_2] = ~torch.ones((section_1, section_2), dtype=torch.bool, device=device).triu(1)
            mask[section_1:section_1 + section_2, section_1:section_1 + section_2] = ~torch.ones((section_2, section_2), dtype=torch.bool, device=device).triu(1)

        mask = torch.where(mask, 0., float('-inf'))
        return mask

    def _sample_next_token(
        self,
        music_sequence: torch.LongTensor,
        motion_sequence: torch.LongTensor,
        cfg_conditions: CFGConditions,
        mode: str,
        use_sampling: bool = False,
        temp: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        cfg_coef: tp.Optional[float] = None,
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        B = music_sequence.shape[0]
        cfg_coef = self.cfg_coef if cfg_coef is None else cfg_coef

        sequence = torch.cat((music_sequence, motion_sequence), dim=-1)
        # get self-attn mask
        src_mask = self.get_self_attn_mask(music_sequence.shape[-1], motion_sequence.shape[-1], mode=mode)
        # get cross-attention mask
        cross_attn_mask = torch.where(cfg_conditions['description'][-1], 0., float('-inf'))

        assert isinstance(cfg_conditions, dict)
        condition_tensors = cfg_conditions
        if condition_tensors:
            # Preparing for CFG, predicting both conditional and unconditional logits.
            sequence = torch.cat([sequence, sequence], dim=0)

        music_all_logits, motion_all_logits = self(
            sequence, conditions=[], condition_tensors=condition_tensors, src_mask=src_mask, cross_attn_mask=cross_attn_mask
        )

        if condition_tensors:
            music_cond_logits, music_uncond_logits = music_all_logits.split(B, dim=0)  # [B, K, T, card]
            motion_cond_logits, motion_uncond_logits = motion_all_logits.split(B, dim=0)  # [B, K, T, card]
            music_logits = music_uncond_logits + (music_cond_logits - music_uncond_logits) * cfg_coef
            motion_logits = motion_uncond_logits + (motion_cond_logits - motion_uncond_logits) * cfg_coef
        else:
            music_logits = music_all_logits
            motion_logits = motion_all_logits

        # sample music tokne
        music_logits = music_logits.permute(0, 1, 3, 2)  # [B, K, card, T]
        music_logits = music_logits[..., -1]  # [B, K, card]
        # Apply softmax for sampling if temp > 0. Else, do greedy sampling to avoid zero division error.
        if use_sampling and temp > 0.0:
            probs = torch.softmax(music_logits / temp, dim=-1)
            if top_p > 0.0:
                music_next_token = utils.sample_top_p(probs, p=top_p)
            elif top_k > 0:
                music_next_token = utils.sample_top_k(probs, k=top_k)
            else:
                music_next_token = utils.multinomial(probs, num_samples=1)
        else:
            music_next_token = torch.argmax(music_logits, dim=-1, keepdim=True)

        # sample music tokne
        motion_logits = motion_logits.permute(0, 1, 3, 2)  # [B, K, card, T]
        motion_logits = motion_logits[..., -1]  # [B, K, card]
        # Apply softmax for sampling if temp > 0. Else, do greedy sampling to avoid zero division error.
        if use_sampling and temp > 0.0:
            probs = torch.softmax(motion_logits / temp, dim=-1)
            if top_p > 0.0:
                motion_next_token = utils.sample_top_p(probs, p=top_p)
            elif top_k > 0:
                motion_next_token = utils.sample_top_k(probs, k=top_k)
            else:
                motion_next_token = utils.multinomial(probs, num_samples=1)
        else:
            motion_next_token = torch.argmax(motion_logits, dim=-1, keepdim=True)

        return music_next_token, motion_next_token

    @torch.no_grad()
    def generate(
        self,
        conditions: tp.List[ConditioningAttributes] = [],
        mode: str = None,
        music_code: tp.Optional[torch.LongTensor] = None,
        motion_code: tp.Optional[torch.LongTensor] = None,
        num_samples: tp.Optional[int] = None,
        max_gen_len: int = 256,
        use_sampling: bool = True,
        temp: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        cfg_coef: tp.Optional[float] = None,
        check: bool = True,
    ) -> tp.Tuple[torch.LongTensor, torch.LongTensor]:
        assert not self.training, "generation shouldn't be used in training mode."
        first_param = next(iter(self.parameters()))
        device = first_param.device

        assert mode in ['music_motion', 'music2motion', 'motion2music']
        assert music_code is None or motion_code is None, "cannot provide both music and motion code."

        # half the condition is music, half is motion
        num_samples = len(conditions) // 2
        assert num_samples > 0

        # get classifier-free guidance conditions
        cfg_conditions: CFGConditions
        if conditions:
            music_conditions = conditions[:num_samples]
            motion_conditions = conditions[num_samples:]
            null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
            null_music_conditions = null_conditions[:num_samples]
            null_motion_conditions = null_conditions[num_samples:]
            music_conditions = music_conditions + null_music_conditions
            motion_conditions = motion_conditions + null_motion_conditions

            tokenized_music = self.condition_provider.tokenize(music_conditions, device)
            condition_tensor_music = self.condition_provider(tokenized_music)
            tokenized_motion = self.condition_provider.tokenize(motion_conditions, device)
            condition_tensor_motion = self.condition_provider(tokenized_motion)

            # merge music and motion conditions
            condition_tensor = torch.cat([
                condition_tensor_music['description'][0], condition_tensor_motion['description'][0]
            ], dim=1)  # [B*2, L_music+L_motion, D]
            # construct cross-attn mask for conditions
            condition_mask = torch.zeros(
                (condition_tensor.shape[0], 2, condition_tensor.shape[-2]),
                dtype=torch.bool, device=device
            )  # [B*2, 2, L_music+L_motion]
            music_condition_mask = condition_tensor_music['description'][1]  # [B*2, L_music]
            motion_condition_mask = condition_tensor_motion['description'][1]  # [B*2, L_motion]
            condition_mask[:, 0, :music_condition_mask.shape[-1]] = music_condition_mask.bool()
            condition_mask[:, 1, music_condition_mask.shape[-1]:] = motion_condition_mask.bool()

            cfg_conditions = {'description': (condition_tensor, condition_mask)}
        else:
            cfg_conditions = {}

        B, K = num_samples, self.num_codebooks

        pattern = self.pattern_provider.get_pattern(max_gen_len)
        # this token is used as default value for codes that are not generated yet
        unknown_token = -1
        # we generate codes up to the max_gen_len that will be mapped to the pattern sequence
        # and replace the unknown code with provided code if necessary
        if music_code is None:
            music_gen_codes = torch.full((B, K, max_gen_len), unknown_token, dtype=torch.long, device=device)
        else:
            music_gen_codes = music_code
        if motion_code is None:
            motion_gen_codes = torch.full((B, K, max_gen_len), unknown_token, dtype=torch.long, device=device)
        else:
            motion_gen_codes = motion_code
        assert music_gen_codes.shape[-1] == motion_gen_codes.shape[-1], "music code and motion code should be in equal time dimension"
        # create the gen_sequence with proper interleaving from the pattern: [B, K, S]
        music_gen_sequence, _, music_mask = pattern.build_pattern_sequence(music_gen_codes, self.special_token_id)   # gen_sequence: padded with self.special_token_id
        motion_gen_sequence, _, motion_mask = pattern.build_pattern_sequence(motion_gen_codes, self.special_token_id)   # gen_sequence: padded with self.special_token_id

        gen_sequence_len = music_gen_sequence.shape[-1]  # gen_sequence shape is [B, K, S]
        for offset in tqdm(range(1, gen_sequence_len), desc=f"Generating music & motion of shape {music_gen_sequence.shape}"):
            # get current sequence
            music_curr_sequence = music_gen_sequence[..., 0:offset]
            music_curr_mask = music_mask[None, ..., 0:offset].expand(B, -1, -1)
            motion_curr_sequence = motion_gen_sequence[..., 0:offset]
            motion_curr_mask = motion_mask[None, ..., 0:offset].expand(B, -1, -1)
            if check:
                # check coherence between mask and sequence
                assert (music_curr_sequence == torch.where(music_curr_mask, music_curr_sequence, self.special_token_id)).all()
                assert (motion_curr_sequence == torch.where(motion_curr_mask, motion_curr_sequence, self.special_token_id)).all()
                # should never happen as gen_sequence is filled progressively
                assert not (music_curr_sequence == unknown_token).any()
                assert not (motion_curr_sequence == unknown_token).any()
            # sample next token from the model, next token shape is [B, K, 1]
            music_next_token, motion_next_token = self._sample_next_token(
                music_curr_sequence, motion_curr_sequence, cfg_conditions, mode, use_sampling,
                temp, top_k, top_p, cfg_coef=cfg_coef
            )
            # ensure the tokens that should be masked are properly set to special_token_id
            # as the model never output special_token_id
            music_valid_mask = music_mask[..., offset:offset+1].expand(B, -1, -1)
            music_next_token[~music_valid_mask] = self.special_token_id
            motion_valid_mask = motion_mask[..., offset:offset + 1].expand(B, -1, -1)
            motion_next_token[~motion_valid_mask] = self.special_token_id
            # We only write over unknown tokens
            # i.e., update the prediction if they are not provided
            if music_code is None:
                music_gen_sequence[..., offset:offset+1] = torch.where(
                    music_gen_sequence[..., offset:offset+1] == unknown_token,
                    music_next_token, music_gen_sequence[..., offset:offset+1]
                )
            if motion_code is None:
                motion_gen_sequence[..., offset:offset + 1] = torch.where(
                    motion_gen_sequence[..., offset:offset + 1] == unknown_token,
                    motion_next_token, motion_gen_sequence[..., offset:offset + 1]
                )

        # ensure sequence has been entirely filled
        assert not (music_gen_sequence == unknown_token).any()
        assert not (motion_gen_sequence == unknown_token).any()
        # ensure gen_sequence pattern and mask are matching
        # which means the gen_sequence is valid according to the pattern
        assert (
            music_gen_sequence == torch.where(music_mask[None, ...].expand(B, -1, -1), music_gen_sequence,
                                              self.special_token_id)).all()
        assert (
            motion_gen_sequence == torch.where(motion_mask[None, ...].expand(B, -1, -1), motion_gen_sequence,
                                               self.special_token_id)).all()
        # get back the codes, trimming the prompt if needed and cutting potentially incomplete timesteps
        music_out_codes, out_indexes, music_out_mask = pattern.revert_pattern_sequence(music_gen_sequence, special_token=unknown_token)
        motion_out_codes, out_indexes, motion_out_mask = pattern.revert_pattern_sequence(motion_gen_sequence, special_token=unknown_token)

        # sanity checks over the returned codes and corresponding masks
        assert (music_out_codes[..., :max_gen_len] != unknown_token).all()
        assert (music_out_mask[..., :max_gen_len] == 1).all()
        assert (motion_out_codes[..., :max_gen_len] != unknown_token).all()
        assert (motion_out_mask[..., :max_gen_len] == 1).all()

        music_out_codes = music_out_codes[..., 0:max_gen_len]
        motion_out_codes = motion_out_codes[..., 0:max_gen_len]

        # ensure the returned codes are all valid
        assert (music_out_codes >= 0).all() and (music_out_codes <= self.card).all()
        assert (motion_out_codes >= 0).all() and (motion_out_codes <= self.card).all()
        return music_out_codes, motion_out_codes

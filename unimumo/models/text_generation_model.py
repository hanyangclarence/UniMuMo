import torch
import torch.nn as nn
from typing import Sequence, List

from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class TextGenerator(nn.Module):
    def __init__(self, model: str = "t5-base", max_length: int = 64, context_dim: int = 1024, self_dim: int = 768):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.add_tokens('<separation>')
        self.max_length = max_length
        self.context_dim = context_dim
        self.self_dim = self_dim

        self.model = T5ForConditionalGeneration.from_pretrained(model)

        self.context_proj = nn.Linear(context_dim, self_dim)

    def get_cross_attn_mask(self, context: torch.Tensor, mode: str):
        device = next(iter(self.parameters())).device
        mask = torch.zeros(context.shape[:2], dtype=torch.int64, device=device)  # [B, L_context]

        assert mode in ['music_caption', 'motion_caption']
        if mode == 'music_caption':
            mask[:, :context.shape[1]//2] = 1
        else:
            mask[:, context.shape[1]//2:] = 1

        return mask

    def forward(self, texts: Sequence[str], music_motion_context: torch.Tensor, mode: str) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True
        )

        device = next(self.model.parameters()).device
        labels = encoded["input_ids"].to(device)
        decoder_attention_mask = encoded["attention_mask"].to(device)

        if any(torch.sum(decoder_attention_mask, dim=-1) > self.max_length):
            print(f'warning!!!!!!!!!, {torch.sum(decoder_attention_mask, dim=-1)}')

        cross_attn_mask = self.get_cross_attn_mask(music_motion_context, mode=mode)
        music_motion_context = self.context_proj(music_motion_context)
        music_motion_context = BaseModelOutput(music_motion_context)

        labels[labels == 0] = -100

        loss = self.model.forward(
            encoder_outputs=music_motion_context,
            attention_mask=cross_attn_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        ).loss

        return loss

    def generate_caption(self, music_motion_context: torch.Tensor, mode: str) -> List[str]:
        cross_attn_mask = self.get_cross_attn_mask(music_motion_context, mode=mode)
        music_motion_context = self.context_proj(music_motion_context)
        music_motion_context = BaseModelOutput(music_motion_context)

        outputs = self.model.generate(
            encoder_outputs=music_motion_context,
            attention_mask=cross_attn_mask,
            do_sample=False,
            max_length=256,
            num_beams=4
        )

        captions = []
        for output in outputs:
            captions.append(self.tokenizer.decode(output, skip_special_tokens=True))

        return captions

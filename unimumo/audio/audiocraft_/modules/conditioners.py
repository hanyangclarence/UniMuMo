# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import chain
import logging
import random
import re
import typing as tp
import warnings

import einops
from num2words import num2words
import spacy
from transformers import RobertaTokenizer, T5EncoderModel, T5Tokenizer  # type: ignore
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .streaming import StreamingModule
from .transformer import create_sin_embedding
from unimumo.audio.audiocraft_.data.audio_dataset import SegmentInfo
from ..utils.autocast import TorchAutocast
from ..utils.utils import collate, hash_trick, length_to_mask


logger = logging.getLogger(__name__)
TextCondition = tp.Optional[str]  # a text condition can be a string or None (if doesn't exist)
ConditionType = tp.Tuple[torch.Tensor, torch.Tensor]  # condition, mask


class WavCondition(tp.NamedTuple):
    wav: torch.Tensor
    length: torch.Tensor
    sample_rate: tp.List[int]
    path: tp.List[tp.Optional[str]] = []
    seek_time: tp.List[tp.Optional[float]] = []


class JointEmbedCondition(tp.NamedTuple):
    wav: torch.Tensor
    text: tp.List[tp.Optional[str]]
    length: torch.Tensor
    sample_rate: tp.List[int]
    path: tp.List[tp.Optional[str]] = []
    seek_time: tp.List[tp.Optional[float]] = []


@dataclass
class ConditioningAttributes:
    text: tp.Dict[str, tp.Optional[str]] = field(default_factory=dict)
    wav: tp.Dict[str, WavCondition] = field(default_factory=dict)
    joint_embed: tp.Dict[str, JointEmbedCondition] = field(default_factory=dict)

    def __getitem__(self, item):
        return getattr(self, item)

    @property
    def text_attributes(self):
        return self.text.keys()

    @property
    def wav_attributes(self):
        return self.wav.keys()

    @property
    def joint_embed_attributes(self):
        return self.joint_embed.keys()

    @property
    def attributes(self):
        return {
            "text": self.text_attributes,
            "wav": self.wav_attributes,
            "joint_embed": self.joint_embed_attributes,
        }

    def to_flat_dict(self):
        return {
            **{f"text.{k}": v for k, v in self.text.items()},
            **{f"wav.{k}": v for k, v in self.wav.items()},
            **{f"joint_embed.{k}": v for k, v in self.joint_embed.items()}
        }

    @classmethod
    def from_flat_dict(cls, x):
        out = cls()
        for k, v in x.items():
            kind, att = k.split(".")
            out[kind][att] = v
        return out


class SegmentWithAttributes(SegmentInfo):
    """Base class for all dataclasses that are used for conditioning.
    All child classes should implement `to_condition_attributes` that converts
    the existing attributes to a dataclass of type ConditioningAttributes.
    """
    def to_condition_attributes(self) -> ConditioningAttributes:
        raise NotImplementedError()


def nullify_condition(condition: ConditionType, dim: int = 1):
    """Transform an input condition to a null condition.
    The way it is done by converting it to a single zero vector similarly
    to how it is done inside WhiteSpaceTokenizer and NoopTokenizer.

    Args:
        condition (ConditionType): A tuple of condition and mask (tuple[torch.Tensor, torch.Tensor])
        dim (int): The dimension that will be truncated (should be the time dimension)
        WARNING!: dim should not be the batch dimension!
    Returns:
        ConditionType: A tuple of null condition and mask
    """
    assert dim != 0, "dim cannot be the batch dimension!"
    assert isinstance(condition, tuple) and \
        isinstance(condition[0], torch.Tensor) and \
        isinstance(condition[1], torch.Tensor), "'nullify_condition' got an unexpected input type!"
    cond, mask = condition
    B = cond.shape[0]
    last_dim = cond.dim() - 1
    out = cond.transpose(dim, last_dim)
    out = 0. * out[..., :1]
    out = out.transpose(dim, last_dim)
    mask = torch.zeros((B, 1), device=out.device).int()
    assert cond.dim() == out.dim()
    return out, mask


def nullify_wav(cond: WavCondition) -> WavCondition:
    """Transform a WavCondition to a nullified WavCondition.
    It replaces the wav by a null tensor, forces its length to 0, and replaces metadata by dummy attributes.

    Args:
        cond (WavCondition): Wav condition with wav, tensor of shape [B, T].
    Returns:
        WavCondition: Nullified wav condition.
    """
    null_wav, _ = nullify_condition((cond.wav, torch.zeros_like(cond.wav)), dim=cond.wav.dim() - 1)
    return WavCondition(
        wav=null_wav,
        length=torch.tensor([0] * cond.wav.shape[0], device=cond.wav.device),
        sample_rate=cond.sample_rate,
        path=[None] * cond.wav.shape[0],
        seek_time=[None] * cond.wav.shape[0],
    )


def nullify_joint_embed(embed: JointEmbedCondition) -> JointEmbedCondition:
    """Nullify the joint embedding condition by replacing it by a null tensor, forcing its length to 0,
    and replacing metadata by dummy attributes.

    Args:
        cond (JointEmbedCondition): Joint embedding condition with wav and text, wav tensor of shape [B, C, T].
    """
    null_wav, _ = nullify_condition((embed.wav, torch.zeros_like(embed.wav)), dim=embed.wav.dim() - 1)
    return JointEmbedCondition(
        wav=null_wav, text=[None] * len(embed.text),
        length=torch.LongTensor([0]).to(embed.wav.device),
        sample_rate=embed.sample_rate,
        path=[None] * embed.wav.shape[0],
        seek_time=[0] * embed.wav.shape[0],
    )


class Tokenizer:
    """Base tokenizer implementation
    (in case we want to introduce more advances tokenizers in the future).
    """
    def __call__(self, texts: tp.List[tp.Optional[str]]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


class WhiteSpaceTokenizer(Tokenizer):
    """This tokenizer should be used for natural language descriptions.
    For example:
    ["he didn't, know he's going home.", 'shorter sentence'] =>
    [[78, 62, 31,  4, 78, 25, 19, 34],
    [59, 77,  0,  0,  0,  0,  0,  0]]
    """
    PUNCTUATION = "?:!.,;"

    def __init__(self, n_bins: int, pad_idx: int = 0, language: str = "en_core_web_sm",
                 lemma: bool = True, stopwords: bool = True) -> None:
        self.n_bins = n_bins
        self.pad_idx = pad_idx
        self.lemma = lemma
        self.stopwords = stopwords
        try:
            self.nlp = spacy.load(language)
        except IOError:
            spacy.cli.download(language)  # type: ignore
            self.nlp = spacy.load(language)

    @tp.no_type_check
    def __call__(self, texts: tp.List[tp.Optional[str]],
                 return_text: bool = False) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Take a list of strings and convert them to a tensor of indices.

        Args:
            texts (list[str]): List of strings.
            return_text (bool, optional): Whether to return text as additional tuple item. Defaults to False.
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Indices of words in the LUT.
                - And a mask indicating where the padding tokens are
        """
        output, lengths = [], []
        texts = deepcopy(texts)
        for i, text in enumerate(texts):
            # if current sample doesn't have a certain attribute, replace with pad token
            if text is None:
                output.append(torch.Tensor([self.pad_idx]))
                lengths.append(0)
                continue

            # convert numbers to words
            text = re.sub(r"(\d+)", lambda x: num2words(int(x.group(0))), text)  # type: ignore
            # normalize text
            text = self.nlp(text)  # type: ignore
            # remove stopwords
            if self.stopwords:
                text = [w for w in text if not w.is_stop]  # type: ignore
            # remove punctuation
            text = [w for w in text if w.text not in self.PUNCTUATION]  # type: ignore
            # lemmatize if needed
            text = [getattr(t, "lemma_" if self.lemma else "text") for t in text]  # type: ignore

            texts[i] = " ".join(text)
            lengths.append(len(text))
            # convert to tensor
            tokens = torch.Tensor([hash_trick(w, self.n_bins) for w in text])
            output.append(tokens)

        mask = length_to_mask(torch.IntTensor(lengths)).int()
        padded_output = pad_sequence(output, padding_value=self.pad_idx).int().t()
        if return_text:
            return padded_output, mask, texts  # type: ignore
        return padded_output, mask


class NoopTokenizer(Tokenizer):
    """This tokenizer should be used for global conditioners such as: artist, genre, key, etc.
    The difference between this and WhiteSpaceTokenizer is that NoopTokenizer does not split
    strings, so "Jeff Buckley" will get it's own index. Whereas WhiteSpaceTokenizer will
    split it to ["Jeff", "Buckley"] and return an index per word.

    For example:
    ["Queen", "ABBA", "Jeff Buckley"] => [43, 55, 101]
    ["Metal", "Rock", "Classical"] => [0, 223, 51]
    """
    def __init__(self, n_bins: int, pad_idx: int = 0):
        self.n_bins = n_bins
        self.pad_idx = pad_idx

    def __call__(self, texts: tp.List[tp.Optional[str]]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        output, lengths = [], []
        for text in texts:
            # if current sample doesn't have a certain attribute, replace with pad token
            if text is None:
                output.append(self.pad_idx)
                lengths.append(0)
            else:
                output.append(hash_trick(text, self.n_bins))
                lengths.append(1)

        tokens = torch.LongTensor(output).unsqueeze(1)
        mask = length_to_mask(torch.IntTensor(lengths)).int()
        return tokens, mask


class BaseConditioner(nn.Module):
    """Base model for all conditioner modules.
    We allow the output dim to be different than the hidden dim for two reasons:
    1) keep our LUTs small when the vocab is large;
    2) make all condition dims consistent.

    Args:
        dim (int): Hidden dim of the model.
        output_dim (int): Output dim of the conditioner.
    """
    def __init__(self, dim: int, output_dim: int):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.output_proj = nn.Linear(dim, output_dim)

    def tokenize(self, *args, **kwargs) -> tp.Any:
        """Should be any part of the processing that will lead to a synchronization
        point, e.g. BPE tokenization with transfer to the GPU.

        The returned value will be saved and return later when calling forward().
        """
        raise NotImplementedError()

    def forward(self, inputs: tp.Any) -> ConditionType:
        """Gets input that should be used as conditioning (e.g, genre, description or a waveform).
        Outputs a ConditionType, after the input data was embedded as a dense vector.

        Returns:
            ConditionType:
                - A tensor of size [B, T, D] where B is the batch size, T is the length of the
                  output embedding and D is the dimension of the embedding.
                - And a mask indicating where the padding tokens.
        """
        raise NotImplementedError()


class TextConditioner(BaseConditioner):
    ...


class LUTConditioner(TextConditioner):
    """Lookup table TextConditioner.

    Args:
        n_bins (int): Number of bins.
        dim (int): Hidden dim of the model (text-encoder/LUT).
        output_dim (int): Output dim of the conditioner.
        tokenizer (str): Name of the tokenizer.
        pad_idx (int, optional): Index for padding token. Defaults to 0.
    """
    def __init__(self, n_bins: int, dim: int, output_dim: int, tokenizer: str, pad_idx: int = 0):
        super().__init__(dim, output_dim)
        self.embed = nn.Embedding(n_bins, dim)
        self.tokenizer: Tokenizer
        if tokenizer == 'whitespace':
            self.tokenizer = WhiteSpaceTokenizer(n_bins, pad_idx=pad_idx)
        elif tokenizer == 'noop':
            self.tokenizer = NoopTokenizer(n_bins, pad_idx=pad_idx)
        else:
            raise ValueError(f"unrecognized tokenizer `{tokenizer}`.")

    def tokenize(self, x: tp.List[tp.Optional[str]]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        device = self.embed.weight.device
        tokens, mask = self.tokenizer(x)
        tokens, mask = tokens.to(device), mask.to(device)
        return tokens, mask

    def forward(self, inputs: tp.Tuple[torch.Tensor, torch.Tensor]) -> ConditionType:
        tokens, mask = inputs
        embeds = self.embed(tokens)
        embeds = self.output_proj(embeds)
        embeds = (embeds * mask.unsqueeze(-1))
        return embeds, mask


class T5Conditioner(TextConditioner):
    """T5-based TextConditioner.

    Args:
        name (str): Name of the T5 model.
        output_dim (int): Output dim of the conditioner.
        finetune (bool): Whether to fine-tune T5 at train time.
        device (str): Device for T5 Conditioner.
        autocast_dtype (tp.Optional[str], optional): Autocast dtype.
        word_dropout (float, optional): Word dropout probability.
        normalize_text (bool, optional): Whether to apply text normalization.
    """
    MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
              "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
              "google/flan-t5-xl", "google/flan-t5-xxl"]
    MODELS_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
    }

    def __init__(self, name: str, output_dim: int, finetune: bool, device: str,
                 autocast_dtype: tp.Optional[str] = 'float32', word_dropout: float = 0.,
                 normalize_text: bool = False):
        assert name in self.MODELS, f"Unrecognized t5 model name (should in {self.MODELS})"
        super().__init__(self.MODELS_DIMS[name], output_dim)
        #self.device = device
        self.name = name
        self.finetune = finetune
        self.word_dropout = word_dropout
        if autocast_dtype is None or device == 'cpu':
            self.autocast = TorchAutocast(enabled=False)
            if device != 'cpu':
                logger.warning("T5 has no autocast, this might lead to NaN")
        else:
            dtype = getattr(torch, autocast_dtype)
            assert isinstance(dtype, torch.dtype)
            print(f"T5 will be evaluated with autocast as {autocast_dtype}")
            self.autocast = TorchAutocast(enabled=True, device_type=device, dtype=dtype)
        # Let's disable logging temporarily because T5 will vomit some errors otherwise.
        # thanks https://gist.github.com/simon-weber/7853144
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                print(f'Load t5 model name: {name}')
                # name = '/home/hanyang/.cache/huggingface/hub/models--t5-base/snapshots/fe6d9bf207cd3337512ca838a8b453f87a9178ef'

                self.t5_tokenizer = T5Tokenizer.from_pretrained(name)

                t5 = T5EncoderModel.from_pretrained(name).train(mode=finetune)
            finally:
                logging.disable(previous_level)
        if finetune:
            self.t5 = t5
        else:
            # this makes sure that the t5 models is not part
            # of the saved checkpoint
            # self.__dict__['t5'] = t5.to(device)
            self.t5 = t5
            for p in self.t5.parameters():
                p.requires_grad = False
            self.t5.eval()

        self.normalize_text = normalize_text
        if normalize_text:
            self.text_normalizer = WhiteSpaceTokenizer(1, lemma=True, stopwords=True)

    def tokenize(self, x: tp.List[tp.Optional[str]], device) -> tp.Dict[str, torch.Tensor]:
        # if current sample doesn't have a certain attribute, replace with empty string
        entries: tp.List[str] = [xi if xi is not None else "" for xi in x]
        if self.normalize_text:
            _, _, entries = self.text_normalizer(entries, return_text=True)
        if self.word_dropout > 0. and self.training:
            new_entries = []
            for entry in entries:
                words = [word for word in entry.split(" ") if random.random() >= self.word_dropout]
                new_entries.append(" ".join(words))
            entries = new_entries

        empty_idx = torch.LongTensor([i for i, xi in enumerate(entries) if xi == ""])
        inputs = self.t5_tokenizer(entries, return_tensors='pt', padding=True).to(device)
        mask = inputs['attention_mask']

        # I remove this. This seems unreasonable
        # mask[empty_idx, :] = 0  # zero-out index where the input is non-existant

        return inputs

    def forward(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:
        mask = inputs['attention_mask']
        with torch.set_grad_enabled(self.finetune), self.autocast:
            embeds = self.t5(**inputs).last_hidden_state
        embeds = self.output_proj(embeds.to(self.output_proj.weight))
        embeds = (embeds * mask.unsqueeze(-1))
        return embeds, mask


def dropout_condition(sample: ConditioningAttributes, condition_type: str, condition: str) -> ConditioningAttributes:
    """Utility function for nullifying an attribute inside an ConditioningAttributes object.
    If the condition is of type "wav", then nullify it using `nullify_condition` function.
    If the condition is of any other type, set its value to None.
    Works in-place.
    """
    if condition_type not in ['text', 'wav', 'joint_embed']:
        raise ValueError(
            "dropout_condition got an unexpected condition type!"
            f" expected 'text', 'wav' or 'joint_embed' but got '{condition_type}'"
        )

    if condition not in getattr(sample, condition_type):
        raise ValueError(
            "dropout_condition received an unexpected condition!"
            f" expected wav={sample.wav.keys()} and text={sample.text.keys()}"
            f" but got '{condition}' of type '{condition_type}'!"
        )

    if condition_type == 'wav':
        wav_cond = sample.wav[condition]
        sample.wav[condition] = nullify_wav(wav_cond)
    elif condition_type == 'joint_embed':
        embed = sample.joint_embed[condition]
        sample.joint_embed[condition] = nullify_joint_embed(embed)
    else:
        sample.text[condition] = ''

    return sample


class DropoutModule(nn.Module):
    """Base module for all dropout modules."""
    def __init__(self, seed: int = 1234):
        super().__init__()
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)


class AttributeDropout(DropoutModule):
    """Dropout with a given probability per attribute.
    This is different from the behavior of ClassifierFreeGuidanceDropout as this allows for attributes
    to be dropped out separately. For example, "artist" can be dropped while "genre" remains.
    This is in contrast to ClassifierFreeGuidanceDropout where if "artist" is dropped "genre"
    must also be dropped.

    Args:
        p (tp.Dict[str, float]): A dict mapping between attributes and dropout probability. For example:
            ...
            "genre": 0.1,
            "artist": 0.5,
            "wav": 0.25,
            ...
        active_on_eval (bool, optional): Whether the dropout is active at eval. Default to False.
        seed (int, optional): Random seed.
    """
    def __init__(self, p: tp.Dict[str, tp.Dict[str, float]], active_on_eval: bool = False, seed: int = 1234):
        super().__init__(seed=seed)
        self.active_on_eval = active_on_eval
        # construct dict that return the values from p otherwise 0
        self.p = {}
        for condition_type, probs in p.items():
            self.p[condition_type] = defaultdict(lambda: 0, probs)

    def forward(self, samples: tp.List[ConditioningAttributes]) -> tp.List[ConditioningAttributes]:
        """
        Args:
            samples (list[ConditioningAttributes]): List of conditions.
        Returns:
            list[ConditioningAttributes]: List of conditions after certain attributes were set to None.
        """
        if not self.training and not self.active_on_eval:
            return samples

        samples = deepcopy(samples)
        for condition_type, ps in self.p.items():  # for condition types [text, wav]
            for condition, p in ps.items():  # for attributes of each type (e.g., [artist, genre])
                if torch.rand(1, generator=self.rng).item() < p:
                    for sample in samples:
                        dropout_condition(sample, condition_type, condition)
        return samples

    def __repr__(self):
        return f"AttributeDropout({dict(self.p)})"


class ClassifierFreeGuidanceDropout(DropoutModule):
    """Classifier Free Guidance dropout.
    All attributes are dropped with the same probability.

    Args:
        p (float): Probability to apply condition dropout during training.
        seed (int): Random seed.
    """
    def __init__(self, p: float, seed: int = 1234):
        super().__init__(seed=seed)
        self.p = p

    def forward(self, samples: tp.List[ConditioningAttributes]) -> tp.List[ConditioningAttributes]:
        """
        Args:
            samples (list[ConditioningAttributes]): List of conditions.
        Returns:
            list[ConditioningAttributes]: List of conditions after all attributes were set to None.
        """
        if not self.training:
            return samples

        # decide on which attributes to drop in a batched fashion
        drop = torch.rand(1, generator=self.rng).item() < self.p
        if not drop:
            return samples

        # nullify conditions of all attributes
        samples = deepcopy(samples)
        for condition_type in ["wav", "text"]:
            for sample in samples:
                for condition in sample.attributes[condition_type]:
                    dropout_condition(sample, condition_type, condition)
        return samples

    def __repr__(self):
        return f"ClassifierFreeGuidanceDropout(p={self.p})"


class ConditioningProvider(nn.Module):
    """Prepare and provide conditions given all the supported conditioners.

    Args:
        conditioners (dict): Dictionary of conditioners.
        device (torch.device or str, optional): Device for conditioners and output condition types.
    """
    def __init__(self, conditioners: tp.Dict[str, BaseConditioner], device: tp.Union[torch.device, str] = "cpu"):
        super().__init__()
        #self.device = device
        self.conditioners = nn.ModuleDict(conditioners)

    @property
    def has_joint_embed_conditions(self):
        return len(self.joint_embed_conditions) > 0

    @property
    def text_conditions(self):
        return [k for k, v in self.conditioners.items() if isinstance(v, TextConditioner)]

    @property
    def has_wav_condition(self):
        return len(self.wav_conditions) > 0

    def tokenize(self, inputs: tp.List[ConditioningAttributes], device) -> tp.Dict[str, tp.Any]:
        """Match attributes/wavs with existing conditioners in self, and compute tokenize them accordingly.
        This should be called before starting any real GPU work to avoid synchronization points.
        This will return a dict matching conditioner names to their arbitrary tokenized representations.

        Args:
            inputs (list[ConditioningAttributes]): List of ConditioningAttributes objects containing
                text and wav conditions.
        """
        assert all([isinstance(x, ConditioningAttributes) for x in inputs]), (
            "Got unexpected types input for conditioner! should be tp.List[ConditioningAttributes]",
            f" but types were {set([type(x) for x in inputs])}"
        )

        output = {}
        text = self._collate_text(inputs)

        assert set(text.keys()).issubset(set(self.conditioners.keys())), (
            f"Got an unexpected attribute! Expected {self.conditioners.keys()}, ",
            f"got {text.keys()}"
        )

        for attribute, batch in text.items():
            output[attribute] = self.conditioners[attribute].tokenize(batch, device)
        return output

    def forward(self, tokenized: tp.Dict[str, tp.Any]) -> tp.Dict[str, ConditionType]:
        """Compute pairs of `(embedding, mask)` using the configured conditioners and the tokenized representations.
        The output is for example:
        {
            "genre": (torch.Tensor([B, 1, D_genre]), torch.Tensor([B, 1])),
            "description": (torch.Tensor([B, T_desc, D_desc]), torch.Tensor([B, T_desc])),
            ...
        }

        Args:
            tokenized (dict): Dict of tokenized representations as returned by `tokenize()`.
        """
        output = {}
        for attribute, inputs in tokenized.items():
            condition, mask = self.conditioners[attribute](inputs)
            output[attribute] = (condition, mask)
        return output

    def _collate_text(self, samples: tp.List[ConditioningAttributes]) -> tp.Dict[str, tp.List[tp.Optional[str]]]:
        """Given a list of ConditioningAttributes objects, compile a dictionary where the keys
        are the attributes and the values are the aggregated input per attribute.
        For example:
        Input:
        [
            ConditioningAttributes(text={"genre": "Rock", "description": "A rock song with a guitar solo"}, wav=...),
            ConditioningAttributes(text={"genre": "Hip-hop", "description": "A hip-hop verse"}, wav=...),
        ]
        Output:
        {
            "genre": ["Rock", "Hip-hop"],
            "description": ["A rock song with a guitar solo", "A hip-hop verse"]
        }

        Args:
            samples (list of ConditioningAttributes): List of ConditioningAttributes samples.
        Returns:
            dict[str, list[str, optional]]: A dictionary mapping an attribute name to text batch.
        """
        out: tp.Dict[str, tp.List[tp.Optional[str]]] = defaultdict(list)
        texts = [x.text for x in samples]
        for text in texts:
            for condition in self.text_conditions:
                out[condition].append(text[condition])
        return out

    def _collate_wavs(self, samples: tp.List[ConditioningAttributes]) -> tp.Dict[str, WavCondition]:
        """Generate a dict where the keys are attributes by which we fetch similar wavs,
        and the values are Tensors of wavs according to said attributes.

        *Note*: by the time the samples reach this function, each sample should have some waveform
        inside the "wav" attribute. It should be either:
        1. A real waveform
        2. A null waveform due to the sample having no similar waveforms (nullified by the dataset)
        3. A null waveform due to it being dropped in a dropout module (nullified by dropout)

        Args:
            samples (list of ConditioningAttributes): List of ConditioningAttributes samples.
        Returns:
            dict[str, WavCondition]: A dictionary mapping an attribute name to wavs.
        """
        wavs = defaultdict(list)
        lengths = defaultdict(list)
        sample_rates = defaultdict(list)
        paths = defaultdict(list)
        seek_times = defaultdict(list)
        out: tp.Dict[str, WavCondition] = {}

        for sample in samples:
            for attribute in self.wav_conditions:
                wav, length, sample_rate, path, seek_time = sample.wav[attribute]
                assert wav.dim() == 3, f"Got wav with dim={wav.dim()}, but expected 3 [1, C, T]"
                assert wav.size(0) == 1, f"Got wav [B, C, T] with shape={wav.shape}, but expected B == 1"
                # mono-channel conditioning
                wav = wav.mean(1, keepdim=True)  # [1, 1, T]
                wavs[attribute].append(wav.flatten())  # [T]
                lengths[attribute].append(length)
                sample_rates[attribute].extend(sample_rate)
                paths[attribute].extend(path)
                seek_times[attribute].extend(seek_time)

        # stack all wavs to a single tensor
        for attribute in self.wav_conditions:
            stacked_wav, _ = collate(wavs[attribute], dim=0)
            out[attribute] = WavCondition(
                stacked_wav.unsqueeze(1), torch.cat(lengths[attribute]), sample_rates[attribute],
                paths[attribute], seek_times[attribute])

        return out

    def _collate_joint_embeds(self, samples: tp.List[ConditioningAttributes]) -> tp.Dict[str, JointEmbedCondition]:
        """Generate a dict where the keys are attributes by which we compute joint embeddings,
        and the values are Tensors of pre-computed embeddings and the corresponding text attributes.

        Args:
            samples (list[ConditioningAttributes]): List of ConditioningAttributes samples.
        Returns:
            A dictionary mapping an attribute name to joint embeddings.
        """
        texts = defaultdict(list)
        wavs = defaultdict(list)
        lengths = defaultdict(list)
        sample_rates = defaultdict(list)
        paths = defaultdict(list)
        seek_times = defaultdict(list)
        channels: int = 0

        out = {}
        for sample in samples:
            for attribute in self.joint_embed_conditions:
                wav, text, length, sample_rate, path, seek_time = sample.joint_embed[attribute]
                assert wav.dim() == 3
                if channels == 0:
                    channels = wav.size(1)
                else:
                    assert channels == wav.size(1), "not all audio has same number of channels in batch"
                assert wav.size(0) == 1, "Expecting single-wav batch in the collate method"
                wav = einops.rearrange(wav, "b c t -> (b c t)")  # [1, C, T] => [C * T]
                wavs[attribute].append(wav)
                texts[attribute].extend(text)
                lengths[attribute].append(length)
                sample_rates[attribute].extend(sample_rate)
                paths[attribute].extend(path)
                seek_times[attribute].extend(seek_time)

        for attribute in self.joint_embed_conditions:
            stacked_texts = texts[attribute]
            stacked_paths = paths[attribute]
            stacked_seek_times = seek_times[attribute]
            stacked_wavs = pad_sequence(wavs[attribute]).to(self.device)
            stacked_wavs = einops.rearrange(stacked_wavs, "(c t) b -> b c t", c=channels)
            stacked_sample_rates = sample_rates[attribute]
            stacked_lengths = torch.cat(lengths[attribute]).to(self.device)
            assert stacked_lengths.size(0) == stacked_wavs.size(0)
            assert len(stacked_sample_rates) == stacked_wavs.size(0)
            assert len(stacked_texts) == stacked_wavs.size(0)
            out[attribute] = JointEmbedCondition(
                text=stacked_texts, wav=stacked_wavs,
                length=stacked_lengths, sample_rate=stacked_sample_rates,
                path=stacked_paths, seek_time=stacked_seek_times)

        return out


class ConditionFuser(StreamingModule):
    """Condition fuser handles the logic to combine the different conditions
    to the actual model input.

    Args:
        fuse2cond (tp.Dict[str, str]): A dictionary that says how to fuse
            each condition. For example:
            {
                "prepend": ["description"],
                "sum": ["genre", "bpm"],
                "cross": ["description"],
            }
        cross_attention_pos_emb (bool, optional): Use positional embeddings in cross attention.
        cross_attention_pos_emb_scale (int): Scale for positional embeddings in cross attention if used.
    """
    FUSING_METHODS = ["sum", "prepend", "cross", "input_interpolate"]

    def __init__(self, fuse2cond: tp.Dict[str, tp.List[str]], cross_attention_pos_emb: bool = False,
                 cross_attention_pos_emb_scale: float = 1.0):
        super().__init__()
        assert all(
            [k in self.FUSING_METHODS for k in fuse2cond.keys()]
        ), f"Got invalid fuse method, allowed methods: {self.FUSING_METHODS}"
        self.cross_attention_pos_emb = cross_attention_pos_emb
        self.cross_attention_pos_emb_scale = cross_attention_pos_emb_scale
        self.fuse2cond: tp.Dict[str, tp.List[str]] = fuse2cond
        self.cond2fuse: tp.Dict[str, str] = {}
        for fuse_method, conditions in fuse2cond.items():
            for condition in conditions:
                self.cond2fuse[condition] = fuse_method

    def forward(
        self,
        input: torch.Tensor,
        conditions: tp.Dict[str, ConditionType]
    ) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """Fuse the conditions to the provided model input.

        Args:
            input (torch.Tensor): Transformer input.
            conditions (dict[str, ConditionType]): Dict of conditions.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: The first tensor is the transformer input
                after the conditions have been fused. The second output tensor is the tensor
                used for cross-attention or None if no cross attention inputs exist.
        """
        B, T, _ = input.shape

        if 'offsets' in self._streaming_state:
            first_step = False
            offsets = self._streaming_state['offsets']
        else:
            first_step = True
            offsets = torch.zeros(input.shape[0], dtype=torch.long, device=input.device)

        assert set(conditions.keys()).issubset(set(self.cond2fuse.keys())), \
            f"given conditions contain unknown attributes for fuser, " \
            f"expected {self.cond2fuse.keys()}, got {conditions.keys()}"
        cross_attention_output = None
        for cond_type, (cond, cond_mask) in conditions.items():
            op = self.cond2fuse[cond_type]
            if op == 'sum':
                input += cond
            elif op == 'input_interpolate':
                cond = einops.rearrange(cond, "b t d -> b d t")
                cond = F.interpolate(cond, size=input.shape[1])
                input += einops.rearrange(cond, "b d t -> b t d")
            elif op == 'prepend':
                if first_step:
                    input = torch.cat([cond, input], dim=1)
            elif op == 'cross':
                if cross_attention_output is not None:
                    cross_attention_output = torch.cat([cross_attention_output, cond], dim=1)
                else:
                    cross_attention_output = cond
            else:
                raise ValueError(f"unknown op ({op})")

        if self.cross_attention_pos_emb and cross_attention_output is not None:
            positions = torch.arange(
                cross_attention_output.shape[1],
                device=cross_attention_output.device
            ).view(1, -1, 1)
            pos_emb = create_sin_embedding(positions, cross_attention_output.shape[-1])
            cross_attention_output = cross_attention_output + self.cross_attention_pos_emb_scale * pos_emb

        if self._is_streaming:
            self._streaming_state['offsets'] = offsets + T

        return input, cross_attention_output

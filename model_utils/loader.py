"""
ref: https://github.com/hiyouga/LLaMA-Factory/blob/v0.8.1/src/llamafactory/model/loader.py
"""
from typing import TYPE_CHECKING, Optional

import torch
from transformers import AutoModelForCausalLM

from llmtuner.hparams.model_args import ModelArguments
from llmtuner.model.loader import _get_init_kwargs, load_config
from llmtuner.model.patcher import patch_config, patch_model
from llmtuner.model.utils.misc import register_autoclass
from llmtuner.model.adapter import init_adapter

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from llmtuner.hparams import FinetuningArguments, ModelArguments


if torch.cuda.device_count() > 1:
    DEVICE_MAP = "auto"
else:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DEVICE_MAP = DEVICE


def extend_load_model(
        tokenizer: "PreTrainedTokenizer",
        model_args: "ModelArguments",
        finetuning_args: "FinetuningArguments",
        is_trainable: bool = False,
        add_valuehead: bool = False,
        device_map: Optional[str] = None,
) -> "PreTrainedModel":
    """扩展 device_map, 重写此方法
    """
    init_kwargs = _get_init_kwargs(model_args)
    if device_map:
        init_kwargs["device_map"] = device_map

    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)

    init_kwargs["config"] = config
    init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path

    print("[DEBUG] load_model init_kwargs: ", init_kwargs)

    model = AutoModelForCausalLM.from_pretrained(**init_kwargs)

    patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
    register_autoclass(config, model, tokenizer)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    model.requires_grad_(False)
    model.eval()

    return model
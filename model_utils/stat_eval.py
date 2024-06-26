"""
扩展 run_sft, 支持多卡推理
ref: https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/train/sft/workflow.py
"""
import yaml

import torch
from transformers import DataCollatorForSeq2Seq

from llmtuner.data import  get_dataset, split_dataset
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.misc import get_logits_processor
from llmtuner.model import load_tokenizer
from llmtuner.train.sft.metric import ComputeMetrics
from llmtuner.train.sft.trainer import CustomSeq2SeqTrainer
from llmtuner.hparams import get_train_args


from .loader import extend_load_model, DEVICE_MAP


def extend_run_sft(generative_args):

    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(generative_args)

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    dataset = get_dataset(model_args, data_args, training_args, stage="sft", **tokenizer_module)

    # 扩展 device_map, 重写此方法
    model = extend_load_model(tokenizer, model_args, finetuning_args, device_map=DEVICE_MAP)

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False if model_args.visual_inputs else training_args.remove_unused_columns

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=None,
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
        **tokenizer_module,
        **split_dataset(dataset, data_args, training_args),
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate:  # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)


def run_step1_task(generative_args) -> None:
    extend_run_sft(generative_args)


if __name__ == '__main__':
    """"""

    with open('yamls/step1.yaml', 'r', encoding='utf-8') as f:
        generative_args = yaml.load(f, Loader=yaml.FullLoader)

    print("[DEBUG] generative_args: ", generative_args)

    extend_run_sft(generative_args)


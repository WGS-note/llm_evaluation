"""
用于通用能力的评估 (mmlu, cmmlu, ceval)
ref: https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/eval/evaluator.py
"""
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict
import yaml
import time
import os
import shutil

import torch
from llmtuner.hparams.model_args import ModelArguments
from llmtuner.hparams import get_eval_args
from llmtuner.model import load_tokenizer
from llmtuner.data import get_template_and_fix_tokenizer
from llmtuner.eval.template import get_eval_template
from llmtuner.extras.constants import CHOICES
from llmtuner.eval.evaluator import Evaluator

from .loader import extend_load_model, DEVICE_MAP


@dataclass
class ExtendedModelArguments(ModelArguments):
    device_map: Optional[str] = field(
        default=None,
        metadata={"help": "Device map for the model, e.g., 'auto'."}
    )


class ExtendedEvaluator(Evaluator):
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:

        self.model_args, self.data_args, self.eval_args, finetuning_args = get_eval_args(args)

        # device_map 扩展模型参数
        # self.model_args = self.extend_model_args(self.model_args)

        self.tokenizer = load_tokenizer(self.model_args)["tokenizer"]
        self.tokenizer.padding_side = "right"  # avoid overflow issue in batched inference for llama2
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args.template)

        # 扩展 device_map, 重写此方法
        # self.model = self._extend_load_model(self.tokenizer, self.model_args, finetuning_args)
        self.model = extend_load_model(self.tokenizer, self.model_args, finetuning_args, device_map=DEVICE_MAP)

        self.eval_template = get_eval_template(self.eval_args.lang)
        self.choice_inputs = [
            self.tokenizer.encode(self.eval_template.prefix + ch, add_special_tokens=False)[-1] for ch in CHOICES
        ]


    def extend_model_args(self, model_args):
        args_dict = asdict(model_args)
        return ExtendedModelArguments(**args_dict)


    def extend_eval(self):
        super().eval()


def run_step3_task(general_args) -> None:
    gl_sss = time.time()
    out_dir = general_args["save_dir"]
    tasks, langs, splits = general_args["tasks"], general_args["langs"], general_args["splits"]
    del general_args["tasks"]
    del general_args["langs"]
    del general_args["splits"]

    for task, lang, split in zip(tasks, langs, splits):

        save_dir = os.path.join(out_dir, task)

        general_args["task"] = task
        general_args["lang"] = lang
        general_args["split"] = split
        general_args["save_dir"] = save_dir

        # evaluator.py 里不支持 overwrite
        if os.path.exists(save_dir) and os.path.isdir(save_dir):
            shutil.rmtree(save_dir)

        print("[DEBUG] task: {} start...".format(task))
        print("[DEBUG] `general_args`: ", general_args)
        sss = time.time()

        evaluator = ExtendedEvaluator(general_args)
        evaluator.extend_eval()

        print("[DEBUG] task: {} end, time: {}".format(task, time.time() - sss))

    print("[DEBUG] general eval end, time: {}".format(time.time() - gl_sss))


if __name__ == '__main__':
    """"""
    # CUDA_VISIBLE_DEVICES=2,3 python ./model_utils/general_evaluator.py

    with open('./yamls/step3.yaml', 'r', encoding='utf-8') as f:
        general_args = yaml.load(f, Loader=yaml.FullLoader)

    print("[DEBUG] general_args: ", general_args)

    run_step3_task(general_args)




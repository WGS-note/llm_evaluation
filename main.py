"""
用于 LLM 的评估

+ 垂域生成能力的统计评估
    - 推理速度
    - 生成能力
        - 统计评估
        - 语义评估
+ 垂域生成能力的客观评估
    - RM (TODO:)
+ 通用生成能力的对比评估
    - 通用能力
    - 角色扮演能力

3 devices
"""
import argparse
import yaml

from model_utils import (run_step1_task,
                         run_step2_task,
                         run_step3_task,
                         run_step4_task)

def parser_args():
    parser = argparse.ArgumentParser(description="llm evaluation")

    parser.add_argument("task", type=str, default="step1")
    parser.add_argument("yaml", type=str, default="./yamls/step1.yaml")

    return parser.parse_args()

args = parser_args()

with open(args.yaml, 'r', encoding='utf-8') as f:
    task_args = yaml.load(f, Loader=yaml.FullLoader)

print("[DEBUG] {} args: {}".format(args.task, task_args))

if args.task == "step1":
    run_step1_task(task_args)
elif args.task == "step2":
    run_step2_task(task_args)
elif args.task == "step3":
    run_step3_task(task_args)
elif args.task == "step4":
    run_step4_task(task_args)

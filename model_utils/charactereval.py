"""
用于角色扮演能力的评估 (CharacterEval)
ref: https://arxiv.org/abs/2401.01275

+ RM 每一项的评分范围是 1-5
"""
import json
import copy
import os
from typing import Dict, Any
import time
import gc
import yaml

from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .baichuanchar_rm.modeling_baichuan import BaichuanCharRM
from .baichuanchar_rm.tokenization_baichuan import BaichuanTokenizer

from .loader import DEVICE_MAP


class CharacterEval():

    @classmethod
    def load_base_model(cls, **kwargs):
        model_path = kwargs.get("model_path")

        if kwargs.get("adapter_name_or_path"):
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=DEVICE_MAP, trust_remote_code=True).eval()
        else:
            from .loader import extend_load_model
            from llmtuner.hparams import get_train_args
            from llmtuner.model import load_tokenizer

            tmp_args = {"model_name_or_path": model_path,
                        "adapter_name_or_path": kwargs.get("adapter_name_or_path"),
                        "template": kwargs.get("template"),
                        "finetuning_type": kwargs.get("finetuning_type")}

            tokenizer_module = load_tokenizer(tmp_args)
            tokenizer = tokenizer_module["tokenizer"]
            model_args, _, _, finetuning_args, _ = get_train_args(tmp_args)
            model = extend_load_model(tokenizer, model_args, finetuning_args, device_map=DEVICE_MAP)

        return cls(tokenizer, model, **kwargs)


    def __init__(self, base_tokenizer, base_model, **kwargs):

        self.tokenizer = base_tokenizer
        self.model = base_model

        self.file_path = kwargs.get("task_dir")
        self.out_path = kwargs.get("output_dir")

        # self._test_data = os.path.join(self.file_path, "test_data_copy.jsonl")
        self._test_data = os.path.join(self.file_path, "test_data.jsonl")
        self._character_profiles = os.path.join(self.file_path, "character_profiles.json")
        self._id2metric = os.path.join(self.file_path, "id2metric.jsonl")

        os.makedirs(self.out_path, exist_ok=True)
        self._check_path_exists()


    def step1_save_base_response(self):
        """CharacterEval step1: 获取base模型的生成结果, 写入目录 out_path
        """
        sss = time.time()
        print("[DEBUG] CharacterEval step1_save_base_response start...")

        with open(self._test_data,'r') as f:
            test_datas = json.load(f)

        # 读取人物介绍
        with open(self._character_profiles,'r') as f:
            self.role_informations = json.load(f)

        results = []
        for data in tqdm(test_datas, total=len(test_datas), desc="Calculating step1_save_base_response"):
            results.append(self._get_llm_response(data))

        self._generation = os.path.join(self.out_path, 'step1_generation.jsonl')
        with open(self._generation,'w') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.flush()

        print("[DEBUG] step1_save_base_response end, time: {}".format(time.time() - sss))


    def step2_transform_format(self):
        """CharacterEval step2: 将上一步的结果转换格式, 供奖励模型评分
        """
        sss = time.time()
        print("[DEBUG] CharacterEval step2_transform_format start...")

        with open(self._id2metric,'r') as f:
            id_metric = json.load(f)

        with open(self._generation,'r') as f:
            datas = json.load(f)

        results = []

        for data in tqdm(datas, total=len(datas), desc="Calculating step2_transform_format"):
            if data['model_output'] is not None and data['model_output'] != "ERROR":
                model_output = data['model_output'].split("\n")[0]  # Prevent continuous generation
                data['model_output'] = model_output
                if str(data['id']) in id_metric:
                    for x in id_metric[str(data['id'])]:
                        data['metric_en']= x[0]
                        data['metric_zh']= x[1]
                        tmp = copy.deepcopy(data)
                        results.append(tmp)

        self._generation_trans = os.path.join(self.out_path, "step2_generation_trans.jsonl")
        with open(self._generation_trans,'w') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.flush()

        print("[DEBUG] step2_transform_format end, time: {}".format(time.time() - sss))


    @torch.no_grad()
    def step3_char_rm(self, reward_model_path="morecry/BaichuanCharRM"):
        """CharacterEval step3: 使用奖励模型对第上一步的结果进行评估
        """
        self._gc_cuda()

        sss = time.time()
        print("[DEBUG] CharacterEval step3_char_rm start...")

        max_seq_length = 4096

        with open(self._character_profiles, "r") as f:
            character_profile = json.load(f)

        with open(self._generation_trans, mode='r') as f:
            records = json.load(f)

        def format_input(example):
            input_text = "<RoleInfo>\n\n" \
                + str(character_profile[example['role']]) + "\n\n<Context>\n\n" + example['context'] + "\n\n<Response>\n\n" + example['model_output'] + "\n\n<Dimension>\n\n" + example["metric_zh"]
            return input_text

        tokenizer = BaichuanTokenizer.from_pretrained(reward_model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        rw_model = BaichuanCharRM.from_pretrained(reward_model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()

        for record in tqdm(records, total=len(records), desc="Calculating step3_char_rm"):
            input_text = format_input(record)
            input_ids = tokenizer.encode(text=input_text, add_special_tokens=False) + [tokenizer.eos_token_id]
            if len(input_ids) > max_seq_length:
                input_ids = input_ids[-max_seq_length:]
            input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()
            with torch.no_grad():
                score = rw_model(input_ids=input_ids)[1].item() * 4 + 1
                record[record['metric_en']] = score

        self._evaluation = os.path.join(self.out_path, 'step3_evaluation.jsonl')
        with open(self._evaluation, 'w') as f:
            f.write(json.dumps(records, ensure_ascii=False, indent=4))

        print("[DEBUG] step3_char_rm end, time: {}".format(time.time() - sss))


    def step4_compute_score(self):
        """CharacterEval step4: 计算最终得分
        """
        sss = time.time()
        print("[DEBUG] CharacterEval step4_compute_score start...")

        score_dict = {}

        with open(self._evaluation, "r") as f:
            records = json.load(f)

        for record in records:
            if record['metric_en'] not in score_dict:
                score_dict[record['metric_en']] = []
            score_dict[record['metric_en']].append(record[record['metric_en']])

        score_dict_log = format_log_data({key: sum(val) / len(val) for key, val in score_dict.items()})

        with open(os.path.join(self.out_path, "step4_eval_scores.jsonl"), "w") as f:
            f.write(score_dict_log)

        print("[DEBUG] step4_compute_score end, time: {}".format(time.time() - sss))


    @torch.no_grad()
    def _get_llm_response(self, data: Dict[str, Any]):

        role = data['role']
        context = data['context']

        role_information = self.role_informations[role]
        role_system = f'''{role_information}
        现在请你扮演一个角色扮演专家。请你根据上述信息扮演{role}进行对话。
        '''

        def __make_inputs(context):
            """处理 context 返回 [{"from":role, "value":dial}, ...]
            """
            dialogues= context.split('\n')
            inputs = []
            for dial in dialogues:
                role = dial.split("：")[0]
                dial = "：".join(dial.split("：")[1:])
                inputs.append({"from":role, "value":dial})

            return inputs

        messages, query = self._concat_messages(__make_inputs(context), role, role_system)

        # "Qwen1.5" 无 chat
        try:
            response, _ = self.model.chat(self.tokenizer, query, history=messages)
        except Exception as e:
            inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
            inputs = inputs.to(self.model.device)
            gen_kwargs = {"do_sample": True, "max_new_tokens": 512}
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        data["model_output"] = response

        return data


    def _concat_messages(self, conversations, role, system):
        history = []
        first_query = system
        if conversations[0]['from'] == role:
            first_response = f"好的！现在我来扮演{role}。" + "我首先发话：" + conversations[0]['value']
        else:
            first_response = f"好的！现在我来扮演{role}。"

        history.append({"role": "user", "content": first_query})
        history.append({"role": "assistant", "content": first_response})

        for i in range(len(conversations)):
            if conversations[i]['from'] == role:
                if i ==0:
                    continue
                else:
                    assert conversations[i-1]['from'] != role
                    query = f"{conversations[i-1]['from']}：" + conversations[i-1]['value']
                    response = f"{conversations[i]['from']}：" + conversations[i]['value']
                history.append({"role": "user", "content": query})
                history.append({"role": "assistant", "content": response})
        assert conversations[-1]['from'] != role

        query = f"{conversations[-1]['from']}：" + conversations[-1]['value']

        return history, query


    def _check_path_exists(self):

        if not os.path.exists(self.file_path) or len(os.listdir(self.file_path)) == 0:
            raise ValueError("{} 不存在或为空".format(self.file_path))

        if not os.path.exists(self._test_data):
            raise ValueError("{} 不存在".format(self._test_data))

        if not os.path.exists(self._character_profiles):
            raise ValueError("{} 不存在".format(self._character_profiles))

        if not os.path.exists(self._id2metric):
            raise ValueError("{} 不存在".format(self._id2metric))


    def _gc_cuda(self):
        del self.tokenizer
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5)


def format_log_data(data_dict: Dict[str, float]):
    """key右对齐保存log
    """

    # 计算最大键和值的长度
    max_key_length = max(len(key) for key in data_dict.keys())
    max_value_length = max(len(f"{value:.2f}") for value in data_dict.values())

    # 格式化字符串
    formatted_lines = []
    for key, value in data_dict.items():
        formatted_line = f"{key:>{max_key_length}}: {value:>{max_value_length}.2f}"
        formatted_lines.append(formatted_line)

    return "\n".join(formatted_lines)


def run_step4_task(chara_args):
    chareval = CharacterEval.load_base_model(**chara_args)

    chareval.step1_save_base_response()
    chareval.step2_transform_format()
    chareval.step3_char_rm(reward_model_path=chara_args.get("reward_model_path"))
    chareval.step4_compute_score()


if __name__ == '__main__':
    """"""
    # CUDA_VISIBLE_DEVICES=2,3 python ./model_utils/charactereval.py

    with open('yamls/step4.yaml', 'r', encoding='utf-8') as f:
        chara_args = yaml.load(f, Loader=yaml.FullLoader)

    print("[DEBUG] chara_args: ", chara_args)

    run_step4_task(chara_args)








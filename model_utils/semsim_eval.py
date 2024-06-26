"""
增加语义维度的评估
ref: https://huggingface.co/thenlper/gte-large-zh
"""
import os
import json
from tqdm import tqdm
import time
import yaml

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from .loader import DEVICE


class SemanticSimilarity():

    @classmethod
    def load_model(cls, model_path, **kwargs):
        model = SentenceTransformer(model_path, device=DEVICE, similarity_fn_name="cosine")
        return cls(model, **kwargs)


    def __init__(self, model, output_dir="./output"):
        self.model = model
        self.file_path = output_dir
        self._generated_predictions = os.path.join(self.file_path, "generated_predictions.jsonl")
        self._predict_results = os.path.join(self.file_path, "predict_results.json")

        if not os.path.exists(self.file_path) or not os.path.exists(self._generated_predictions) or not os.path.exists(self._predict_results):
            raise ValueError("确保 {} 下有 {} 和 {}".format(self.file_path, self._generated_predictions, self._predict_results))


    def run_sen_sim(self,):
        """垂域生成能力的统计评估: 语义相似

        依赖上一步的结果: generated_predictions.jsonl、predict_results.json。 Defaults to "./output"
        """
        sss = time.time()
        print("[DEBUG] run_sen_sim start...")

        data = []
        scores = []
        scores_lst = []

        with open(self._generated_predictions, 'r', encoding='utf-8') as f:
            for line in f:
                json_obj = json.loads(line.strip())
                data.append(json_obj)

        for item in tqdm(data, total=len(data), desc="Calculating similarities"):
            score = self._get_cos_sim_(item['label'], item['predict'])
            scores.append(score)

            item["score"] = float(score)
            scores_lst.append(item)

        # 将结果写入 generated_predictions.jsonl
        with open(self._generated_predictions, 'w', encoding='utf-8') as f:
            for item in scores_lst:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + '\n')

        # 将结果写入 predict_results.json
        with open(self._predict_results, 'r', encoding='utf-8') as f:
            data = json.load(f)

        data["avg_sem_sim_score"] = float(np.mean(scores))
        with open(self._predict_results, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=4))

        print("[DEBUG] run_sen_sim end, time: ", time.time() - sss)


    def _get_cos_sim_(self, str1, str2):
        embeddings = self.model.encode([str1, str2])
        return cos_sim(embeddings[0], embeddings[1]).flatten().numpy()[0]


def run_step2_task(semsim_args) -> None:
    sensim = SemanticSimilarity.load_model(**semsim_args)
    sensim.run_sen_sim()


if __name__ == '__main__':
    """"""

    with open('yamls/step2.yaml', 'r', encoding='utf-8') as f:
        semsim_args = yaml.load(f, Loader=yaml.FullLoader)

    print("[DEBUG] semsim_args: ", semsim_args)

    sensim = SemanticSimilarity.load_model(**semsim_args)
    sensim.run_sen_sim()


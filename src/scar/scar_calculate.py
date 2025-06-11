import sys, os, json, torch
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/SCAR_data_description/")
import numpy as np

from scipy.stats import truncnorm, norm
from scipy.optimize import minimize
from scipy.spatial.distance import jensenshannon


class SCARcalculation():
    def __init__(self, ):
        pass

    def estimate_jsd_from_positive_samples(self, samples, bins=100):
        samples = np.asarray(samples)
        if np.any(samples < 0) or len(samples) < 1:
            raise ValueError("All samples must be positive and larger than 2.")

        # 计算样本均值和无偏样本方差（除以 n-1）
        mu = np.mean(samples)
        sigma = np.std(samples, ddof=1)  # ddof=1 表示无偏估计

        # 样本分布直方图离散化，概率归一化
        hist, bin_edges = np.histogram(samples, bins=bins, density=True)
        p = hist + 1e-12  # 避免零概率
        p /= np.sum(p)

        # 在同样的 bin 中计算正态分布的概率密度
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        q = norm.pdf(bin_centers, loc=mu, scale=sigma) + 1e-12
        q /= np.sum(q)

        # 计算Jensen-Shannon距离的平方（0-1之间）
        jsd = jensenshannon(p, q, base=2) ** 2
        return jsd

    def format_as_binary_per_class(self, logits_cpu, preds_cpu, labels_cpu, num_classes):
        logits_cpu = np.array(logits_cpu)
        preds_cpu = np.array(preds_cpu)
        labels_cpu = np.array(labels_cpu)

        output_per_class = {}

        for c in range(num_classes):
            class_logits_1d = logits_cpu[:, c]

            binary_logits = np.zeros((len(class_logits_1d), 2))
            class_preds = np.zeros(len(class_logits_1d))

            for i, (logit_val, label_val) in enumerate(zip(class_logits_1d, labels_cpu == c)):
                pred_class = np.argmax(logits_cpu[i])

                if not label_val:  # 负类
                    binary_logits[i, 0] = -logit_val
                    binary_logits[i, 1] = 0
                    class_preds[i] = (pred_class != c).astype(int)
                else:  # 正类
                    binary_logits[i, 0] = 0
                    binary_logits[i, 1] = logit_val
                    class_preds[i] = (pred_class == c).astype(int)

            class_labels = (labels_cpu == c).astype(int)

            output_per_class[c] = [
                binary_logits,
                class_preds,
                class_labels
            ]

        return output_per_class
    
    def calculate_scar(self, results_list, num_samples, ratio):
        task_names = list(results_list.keys())

        all_positives = np.array([results_list[key][1] for key in results_list])
        all_negative_mask = np.all(all_positives == 0, axis=0)
        all_negative_indices = np.where(all_negative_mask)[0]

        metrics_per_task = {}
        for task in task_names:
            logits, preds_cpu, labels = results_list[task]

            logits = np.array(logits)
            preds_cpu = np.array(preds_cpu)
            labels = np.array(labels)

            valid_indices = np.setdiff1d(np.arange(num_samples), all_negative_indices)
            true_indices = valid_indices[preds_cpu[valid_indices] == True]    # valid index
            false_indices = valid_indices[preds_cpu[valid_indices] == False]         # plausald index

            other_tasks = [t for t in task_names if t != task]

            # --- Task Related Index: Coverage & Authenticity ---
            k = logits.shape[1]
            coverage_values = []
            for c in range(k):
                mask = (labels == c) & (preds_cpu == 1)
                class_logits = logits[mask, c]
                positive_logits = class_logits[class_logits > 0]
                if len(positive_logits) > 0:
                    coverage_cur = 1 - self.estimate_jsd_from_positive_samples(positive_logits)
                    coverage_values.append(coverage_cur)

            authenticity_value = len(true_indices) / num_samples

            # --- Task non-Related Index: Scale & Richness ---
            scale = (num_samples - len(all_negative_indices)) / num_samples
            scale_value = scale / ratio

            richness_values = [1 for _ in range(len(true_indices))]
            for idx in false_indices:
                other_corrects = [results_list[other][1][idx] for other in other_tasks]
                richness = np.mean(other_corrects)
                richness_values.append(richness)

            # ------------------------------------------

            metrics_per_task[task] = {
                "scale": scale_value,
                "coverage": float(np.mean(coverage_values)) if len(coverage_values) > 0 else 0.0,
                "authenticity": authenticity_value,
                "richness": float(np.mean(richness_values)) if len(richness_values) > 0 else 0.0
            }

        return metrics_per_task

        # all_scales = []
        # all_coverages = []
        # all_authenticities = []
        # all_richnesses = []

        # print("Per-task SCAR components:")
        # for task, metrics in metrics_per_task.items():
        #     scale = metrics["scale"]
        #     coverage = metrics["coverage"]
        #     authenticity = metrics["authenticity"]
        #     richness = metrics["richness"]

        #     print(f"{task:<10} Scale: {scale:.4f}, Coverage: {coverage:.4f}, Authenticity: {authenticity:.4f}, Richness: {richness:.4f}")

        #     all_scales.append(scale)
        #     all_coverages.append(coverage)
        #     all_authenticities.append(authenticity)
        #     all_richnesses.append(richness)

        # scar_index = {
        #     "scale": np.mean(all_scales),
        #     "coverage": np.mean(all_coverages),
        #     "authenticity": np.mean(all_authenticities),
        #     "richness": np.mean(all_richnesses)
        # }

        # return scar_index


    def calculation(self, results_list, num_samples, ratio):
        task_names = list(results_list.keys())

        # ———————————————————— 不同的任务 ——————————————————————
        task_scar = self.calculate_scar(results_list, num_samples, ratio)

        metrics_per_task = {}
        for task in task_names:             
            # —————————————— 一个任务下的不同信号函数 ————————————————
            logits, preds_cpu, labels = results_list[task]

            num_classes = logits[0].shape[0]
            output_binary_class = self.format_as_binary_per_class(logits, preds_cpu, labels, num_classes)

            hs_scar = self.calculate_scar(output_binary_class, num_samples, ratio)

            metrics_per_task[task] = {
                "task_scar" : task_scar[task],
                "hs_scar" : hs_scar
            }

        return metrics_per_task
    
    def count_h(self, s, c, a, r):
        r, delta, err_gen, err_emp = s, 1-c, 1-a, 1-r
        h = delta * np.exp(2 * r * (err_gen - err_emp + 1e-6) ** 2) 

        return h
    
    def fit_lower_exp_function(self, x_list, y_list, lambda_range=(1e-6, 10.0), n_search=100):
        x = np.array(x_list)
        y = np.array(y_list)

        if np.any((x <= 0) | (y <= 0)):
            raise ValueError("x 和 y 必须为正值。")

        best_a = 0
        best_lambda = None

        lambda_vals = np.logspace(np.log10(lambda_range[0]), np.log10(lambda_range[1]), n_search)

        for lam in lambda_vals:
            f = 1 - np.exp(-lam * x)
            if np.any(f <= 0):
                continue

            a_max = np.min(y / f)             # 使得 a * f(x_i) <= y_i ⇒ a <= y_i / f(x_i)

            if a_max > best_a:
                best_a = a_max
                best_lambda = lam

        if best_lambda is None:
            raise RuntimeError("找不到任何合法的 (a, λ) 组合，使函数位于所有点下方。")

        return best_a, best_lambda

    def predict_foudation(self, ratios, indexes, max_h=-1):
        scals, coves, auths, richs = indexes
        hs = [self.count_h(s, c, a, r) for s, c, a, r in zip(scals, coves, auths, richs)]

        cur_h, lambd = self.fit_lower_exp_function(ratios, hs)  

        h = max(cur_h, max_h)

        delta = 0.0001
        epsilon = 0.01      # general - empirical

        foundation_size = np.log(h/(delta)) / (2 * (epsilon) ** 2)


        return cur_h, foundation_size

if __name__ == "__main__":
    example = {
        "what": [
            [  # logits（每个为 np.array）
                np.array([1.2, 0.3], dtype=np.float32),
                np.array([0.1, 2.4], dtype=np.float32),
                np.array([2.1, -0.5], dtype=np.float32),
                np.array([-1.0, 0.8], dtype=np.float32)
            ],
            [  # 是否分类正确
                True,
                True,
                False,
                True
            ],
            [  # 标签
                0,
                1,
                1,
                1
            ]
        ],
        "where": [
            [
                np.array([-0.6, 1.0, 0.2], dtype=np.float32),
                np.array([0.4, -0.8, 0.9], dtype=np.float32),
                np.array([1.5, 0.3, -0.7], dtype=np.float32),
                np.array([0.2, -1.1, 1.3], dtype=np.float32)
            ],
            [
                False,
                True,
                True,
                False
            ],
            [
                2,
                0,
                1,
                2
            ]
        ]
    }

    def format_nested_dict(d, indent=0):
        for key, value in d.items():
            prefix = "  " * indent
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                format_nested_dict(value, indent + 1)
            else:
                if isinstance(value, float):
                    value = f"{value:.4f}"
                print(f"{prefix}  {key}: {value}")

    calculation = SCARcalculation()
    metrics_per_task = calculation.calculation(example, 4, 2)
    print(metrics_per_task)
    format_nested_dict(metrics_per_task)

    # result_list = [{'what': {'task_scar': {'scale': 0.5, 'coverage': 0.9987034797668457, 'authenticity': 0.9898, 'richness': 0.99954}, 'hs_scar': {0: {'scale': 0.5, 'coverage': 0.8058268323019196, 'authenticity': 0.0102, 'richness': 1.0}, 1: {'scale': 0.5, 'coverage': 0.9986024938059566, 'authenticity': 0.9898, 'richness': 1.0}}}, 'where': {'task_scar': {'scale': 0.5, 'coverage': 0.985785186290741, 'authenticity': 0.97684, 'richness': 0.99918}, 'hs_scar': {0: {'scale': 0.5, 'coverage': 0.9968088205924266, 'authenticity': 0.02316, 'richness': 0.51158}, 1: {'scale': 0.5, 'coverage': 0.9984319175618015, 'authenticity': 0.97684, 'richness': 0.98842}, 2: {'scale': 0.5, 'coverage': 0.0, 'authenticity': 0.0, 'richness': 0.5}}}, 'how': {'task_scar': {'scale': 0.5, 'coverage': 0.9851251840591431, 'authenticity': 0.93832, 'richness': 0.999}, 'hs_scar': {0: {'scale': 0.5, 'coverage': 0.9946153896672563, 'authenticity': 0.06168, 'richness': 0.3744533333333333}, 1: {'scale': 0.5, 'coverage': 0.9982770853667962, 'authenticity': 0.93832, 'richness': 0.9588799999999998}, 2: {'scale': 0.5, 'coverage': 0.0, 'authenticity': 0.0, 'richness': 0.33333333333333326}, 3: {'scale': 0.5, 'coverage': 0.0, 'authenticity': 0.0, 'richness': 0.33333333333333326}}}}, {'what': {'task_scar': {'scale': 0.5, 'coverage': 0.9991136789321899, 'authenticity': 0.9896, 'richness': 0.99936}, 'hs_scar': {0: {'scale': 0.5, 'coverage': 0.8697561996712722, 'authenticity': 0.0104, 'richness': 1.0}, 1: {'scale': 0.5, 'coverage': 0.997711581412311, 'authenticity': 0.9896, 'richness': 1.0}}}, 'where': {'task_scar': {'scale': 0.5, 'coverage': 0.9933228492736816, 'authenticity': 0.97672, 'richness': 0.9987}, 'hs_scar': {0: {'scale': 0.5, 'coverage': 0.998922189218729, 'authenticity': 0.02328, 'richness': 0.51164}, 1: {'scale': 0.5, 'coverage': 0.9951632929203973, 'authenticity': 0.97672, 'richness': 0.98836}, 2: {'scale': 0.5, 'coverage': 0.0, 'authenticity': 0.0, 'richness': 0.5}}}, 'how': {'task_scar': {'scale': 0.5, 'coverage': 0.9815216064453125, 'authenticity': 0.9282, 'richness': 0.99834}, 'hs_scar': {0: {'scale': 0.5, 'coverage': 0.9972034633714831, 'authenticity': 0.0718, 'richness': 0.38119999999999993}, 1: {'scale': 0.5, 'coverage': 0.9976408215988579, 'authenticity': 0.9282, 'richness': 0.9521333333333333}, 2: {'scale': 0.5, 'coverage': 0.0, 'authenticity': 0.0, 'richness': 0.33333333333333326}, 3: {'scale': 0.5, 'coverage': 0.0, 'authenticity': 0.0, 'richness': 0.33333333333333326}}}}]
    # results = {'what': {'task_scar': {'scale': 0.5, 'coverage': 0.9989085793495178, 'authenticity': 0.9897, 'richness': 0.99945}, 'hs_scar': {0: {'scale': 0.5, 'coverage': 0.8377915159865958, 'authenticity': 0.0103, 'richness': 1.0}, 1: {'scale': 0.5, 'coverage': 0.9981570376091338, 'authenticity': 0.9897, 'richness': 1.0}}}, 'where': {'task_scar': {'scale': 0.5, 'coverage': 0.9895540177822113, 'authenticity': 0.97678, 'richness': 0.9989399999999999}, 'hs_scar': {0: {'scale': 0.5, 'coverage': 0.9978655049055778, 'authenticity': 0.023219999999999998, 'richness': 0.51161}, 1: {'scale': 0.5, 'coverage': 0.9967976052410994, 'authenticity': 0.97678, 'richness': 0.98839}, 2: {'scale': 0.5, 'coverage': 0.0, 'authenticity': 0.0, 'richness': 0.5}}}, 'how': {'task_scar': {'scale': 0.5, 'coverage': 0.9833233952522278, 'authenticity': 0.93326, 'richness': 0.99867}, 'hs_scar': {0: {'scale': 0.5, 'coverage': 0.9959094265193698, 'authenticity': 0.06674, 'richness': 0.37782666666666664}, 1: {'scale': 0.5, 'coverage': 0.997958953482827, 'authenticity': 0.93326, 'richness': 0.9555066666666665}, 2: {'scale': 0.5, 'coverage': 0.0, 'authenticity': 0.0, 'richness': 0.33333333333333326}, 3: {'scale': 0.5, 'coverage': 0.0, 'authenticity': 0.0, 'richness': 0.33333333333333326}}}}

    # for res in result_list:
    #     format_nested_dict(res)
    
    # print("Final")
    # format_nested_dict(results)
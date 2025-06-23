import sys, os, json, torch
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/SCAR_data_description/")
import numpy as np

from scipy.stats import norm
from scipy.optimize import fsolve, brentq
from itertools import combinations
from scipy.special import logsumexp
from scipy.spatial.distance import jensenshannon


class SCARcalculation():
    def __init__(self, ):
        pass

    def calculation(self, results_list, num_samples, ratio):
        "基于测试结果计算每个 step function 单独的 SCAR 指标。"
        task_names = list(results_list.keys())
        metrics_per_task = {}
        for task in task_names:             
            # —————————————— 单个任务下的不同信号函数 ————————————————
            train_logits, train_preds, train_labels = results_list[task]
            num_classes = train_logits.shape[1]

            output_binary_class = self.format_as_binary_per_class(train_logits, train_preds, train_labels, num_classes)
            hs_scar = self.calculate_scar(output_binary_class, num_samples, ratio)

            metrics_per_task[task] = hs_scar

        return metrics_per_task

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
        output_per_class = {}

        for c in range(num_classes):
            class_logits_1d = logits_cpu[:, c]

            binary_logits = np.zeros((len(class_logits_1d), 2))
            class_preds = np.zeros(len(class_logits_1d))

            for i, (logit, pred, label) in enumerate(zip(class_logits_1d, preds_cpu, labels_cpu == c)):
                if not label:  # 负类
                    binary_logits[i, 0] = -logit
                    binary_logits[i, 1] = 0
                    class_preds[i] = (pred != c).astype(int)
                else:  # 正类
                    binary_logits[i, 0] = 0
                    binary_logits[i, 1] = logit
                    class_preds[i] = (pred == c).astype(int)
            class_labels = (labels_cpu == c).astype(int)

            output_per_class[c] = [binary_logits, class_preds, class_labels]

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
    
    def count_h(self, s, c, a, r):
        r, delta, err_gen, err_emp = s, 1-c, 1-a, 1-r
        h = delta * np.exp(2 * r * (err_gen - err_emp) ** 2) 

        return h
    
    def fit_lower_exp_function(self, x_list, y_list, lambda_range=(1e-4, 10.0), n_search=1000):
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
            print(x_list)
            print(y_list)
            raise RuntimeError("找不到任何合法的 (a, λ) 组合，使函数位于所有点下方。")

        return best_a, best_lambda

    def predict_foudation_step(self, sample_num, class_num, ratios, indexes):
        scals, coves, auths, richs = indexes

        xs = [1/item for item in ratios]
        hs = [self.count_h(s, c, a, r) for s, c, a, r in zip(scals, coves, auths, richs)]

        h, lambd = self.fit_lower_exp_function(xs, hs)  

        delta = 0.01
        # epsilon = 0.001      # general - empirical
        epsilon = 1 - (0.99) ** (1/class_num)

        foundation_size = np.log(h/(delta)) / (2 * (epsilon) ** 2)

        return h, foundation_size

    def predict_foundation_total(self, H_list, delta_E=0.01, epsilon=0.01, verbose=False):
        H_list = np.array(H_list)
        k = len(H_list)
        
        if np.any(H_list <= 0):
            raise ValueError("All h_j must be > 0")

        logsumh = np.log(np.sum(H_list))              # log(∑h_j)
        logprodh = np.sum(np.log(H_list))             # log(∏h_j)
        log_k_minus_1 = np.log(k - 1)

        def f(x):  # x = n * ε²
            logterm1 = -2 * x + logsumh
            logterm2 = log_k_minus_1 - 2 * k * x + logprodh
            M = max(logterm1, logterm2)                 # log-sum-exp trick 

            # log-domain subtraction
            diff = np.exp(logterm1 - M) - np.exp(logterm2 - M)
            return np.exp(M) * diff - delta_E

        # Initial interval for x = n * ε²
        x_min = np.log(np.max(H_list)) / 2 + 1e-3  # ensure t * h_j < 1
        x_max = x_min + 30

        try:
            x_solution = brentq(f, x_min, x_max)
            n = x_solution / epsilon**2
            if verbose:
                print(f"Solution x = {x_solution}, n = {n}")
            return n
        except Exception as e:
            if verbose:
                print("❌ Solve failed:", e)
            return None
        
    def predict_foundation_total_general(self, H_list, delta_E=0.01, epsilon=0.001, verbose=False, max_order=None):
        H_list = np.array(H_list)
        k = len(H_list)

        if np.any(H_list <= 0):
            raise ValueError("All h_j must be > 0")

        if max_order is None:
            max_order = k  # default: use full Bonferroni expansion

        # 预计算 log(h_j)
        log_h_list = np.log(H_list)

        # 预计算每个阶数 r 的 log S_r
        log_S_terms = []
        for r in range(1, max_order + 1):
            log_products = []
            for idxs in combinations(range(k), r):
                log_prod = np.sum(log_h_list[list(idxs)])  # log(∏ h_j)
                log_products.append(log_prod)
            log_S_r = logsumexp(log_products) if log_products else -np.inf
            log_S_terms.append(log_S_r)

        signs = np.array([(-1)**(r+1) for r in range(1, max_order + 1)])  # (-1)^{r+1}

        def f(x):  # x = n * ε²
            log_terms = np.array([-2 * (r + 1) * x + log_S_terms[r] for r in range(max_order)])
            max_log_term = np.max(log_terms)
            signed_sum = np.sum(signs * np.exp(log_terms - max_log_term))
            return np.exp(max_log_term) * signed_sum - delta_E

        x_min = np.log(np.max(H_list)) / 2 + 1e-3
        x_max = x_min + 30

        try:
            x_solution = brentq(f, x_min, x_max)
            n = x_solution / (epsilon)**2
            if verbose:
                print(f"Solution x = {x_solution}, n = {n}")
            return n
        except Exception as e:
            if verbose:
                print("❌ Solve failed:", e)
            return None
    

    

if __name__ == "__main__":
    calculation = SCARcalculation()
    with open("src/scar/hs_list1.json", "r") as f:
        hs_list = json.load(f)

    result = calculation.predict_foundation_total(hs_list, epsilon=0.01)
    print(result)
    
    result = calculation.predict_foundation_total_general(hs_list, epsilon=0.01)
    print(result)


# (1 - (0.999) ** 10) ** (1 / 10)
# 1- 0.99 ** (1 / 10)
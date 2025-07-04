import sys, os, json, torch, random, time
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/SCAR_data_description/")
import numpy as np
from joblib import Parallel, delayed

from scipy.stats import norm
from scipy.optimize import fsolve, brentq
from itertools import combinations
from scipy.special import logsumexp
from scipy.spatial.distance import jensenshannon


class SCARcalculation():
    def __init__(self, ):
        pass

    def calculation(self, results_list, num_samples, ratio):
        task_names = list(results_list.keys())
        metrics_per_task = {}

        for task in task_names:
            train_logits, train_preds, train_labels = results_list[task]
            num_classes = train_logits.shape[1]
            
            output_binary_class = self.format_as_binary_per_class(train_logits, train_preds, train_labels, num_classes)
            hs_scar = self.calculate_scar(output_binary_class, num_samples, ratio)

            accuracy = (train_preds == train_labels).sum() / len(train_labels)

            metrics_per_task[task] = hs_scar

        return metrics_per_task, accuracy

    def format_as_binary_per_class(self, logits_cpu, preds_cpu, labels_cpu, num_classes):
        output_per_class = {}
        n_samples = logits_cpu.shape[0]

        is_multihot = labels_cpu.ndim == 2

        for c in range(num_classes):
            class_logits = logits_cpu[:, c]
            binary_logits = np.zeros((n_samples, 2))

            if is_multihot:
                class_labels = labels_cpu[:, c].astype(int)
            else:
                class_labels = (labels_cpu == c).astype(int)

            binary_logits[:, 0] = -class_logits * (1 - class_labels)
            binary_logits[:, 1] =  class_logits * class_labels

            if is_multihot:
                class_preds = (preds_cpu[:, c] == 1).astype(int)
            else:
                class_preds = (preds_cpu == c).astype(int)

            output_per_class[c] = [binary_logits, class_preds, class_labels]

        return output_per_class

    # def calculate_scar(self, results_list, num_samples, ratio):
    #     task_names = list(results_list.keys())

    #     all_positives = np.array([results_list[key][1] for key in results_list])
    #     all_negative_mask = np.all(all_positives == 0, axis=0)
    #     all_negative_indices = np.where(all_negative_mask)[0]

    #     metrics_per_task = {}
    #     for task in task_names:
    #         logits, preds_cpu, labels = results_list[task]

    #         logits = np.array(logits)
    #         preds_cpu = np.array(preds_cpu)
    #         labels = np.array(labels)

    #         valid_indices = np.setdiff1d(np.arange(num_samples), all_negative_indices)
    #         true_indices = valid_indices[preds_cpu[valid_indices] == True]    # valid index
    #         false_indices = valid_indices[preds_cpu[valid_indices] == False]         # plausald index

    #         other_tasks = [t for t in task_names if t != task]

    #         # --- Task Related Index: Coverage & Authenticity ---
    #         k = logits.shape[1]
    #         coverage_values = []
    #         for c in range(k):
    #             mask = (labels == c) & (preds_cpu == 1)
    #             class_logits = logits[mask, c]
    #             positive_logits = class_logits[class_logits > 0]
    #             if len(positive_logits) > 0:
    #                 coverage_cur = 1 - self.estimate_jsd_from_positive_samples(positive_logits)
    #                 coverage_values.append(coverage_cur)

    #         authenticity_value = len(true_indices) / num_samples

    #         # --- Task non-Related Index: Scale & Richness ---
    #         scale = (num_samples - len(all_negative_indices)) / num_samples
    #         scale_value = scale / ratio

    #         richness_values = [1 for _ in range(len(true_indices))]
    #         for idx in false_indices:
    #             other_corrects = [results_list[other][1][idx] for other in other_tasks]
    #             richness = np.mean(other_corrects)
    #             richness_values.append(richness)

    #         # ------------------------------------------

    #         metrics_per_task[task] = {
    #             "scale": scale_value,
    #             "coverage": float(np.mean(coverage_values)) if len(coverage_values) > 0 else 0.0,
    #             "authenticity": authenticity_value,
    #             "richness": float(np.mean(richness_values)) if len(richness_values) > 0 else 0.0
    #         }

    #     return metrics_per_task

    def estimate_jsd_fast(self, samples, bins=100):
        samples = np.asarray(samples)
        if np.any(samples < 0) or len(samples) < 1:
            return 1.0
        mu = np.mean(samples)
        sigma = np.std(samples, ddof=1)
        if sigma == 0 or np.isnan(sigma):
            return 1.0
        hist, bin_edges = np.histogram(samples, bins=bins, density=True)
        p = hist + 1e-12
        p /= np.sum(p)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        q = norm.pdf(bin_centers, loc=mu, scale=sigma) + 1e-12
        q /= np.sum(q)
        jsd = jensenshannon(p, q, base=2) ** 2
        return jsd

    def calculate_scar(self, results_list, num_samples, ratio):
        task_names = list(results_list.keys())
        k = results_list[task_names[0]][0].shape[1]  # 分类数

        all_preds = np.array([results_list[t][1] for t in task_names])  # (T, N)
        all_negative_mask = np.all(all_preds == 0, axis=0)
        all_negative_indices = np.where(all_negative_mask)[0]
        valid_mask_global = ~all_negative_mask

        metrics_per_task = {}

        for task in task_names:
            logits, preds_cpu, labels = map(np.array, results_list[task])
            positive_mask = preds_cpu == 1
            valid_mask = valid_mask_global.copy()

            true_mask = valid_mask & positive_mask
            false_mask = valid_mask & (preds_cpu == 0)
            true_count = np.sum(true_mask)
            false_indices = np.where(false_mask)[0]

            # --- Coverage ---
            def get_jsd_for_class(c):
                class_mask = (labels == c) & positive_mask
                if np.count_nonzero(class_mask) <= 1:
                    return None
                pos_logits = logits[class_mask, c]
                pos_logits = pos_logits[pos_logits > 0]
                if len(pos_logits) <= 1:
                    return None
                jsd = self.estimate_jsd_fast(pos_logits, bins=100)
                return 1 - jsd

            unique_classes = np.unique(labels)
            coverage_results = Parallel(n_jobs=-1)(
                delayed(get_jsd_for_class)(c) for c in unique_classes
            )
            coverage_values = [v for v in coverage_results if v is not None]
            coverage = float(np.mean(coverage_values)) if coverage_values else 0.0

            # --- Authenticity ---
            authenticity = true_count / num_samples

            # --- Scale ---
            scale = (num_samples - len(all_negative_indices)) / (num_samples * ratio)

            # --- Richness ---
            other_tasks = [t for t in task_names if t != task]
            other_preds = np.array([results_list[other][1][false_indices] for other in other_tasks])  # (T-1, |F|)
            if other_preds.size > 0:
                richness_false = np.mean(other_preds, axis=0)
                richness_values = np.concatenate([
                    np.ones(true_count),
                    richness_false
                ])
            else:
                richness_values = np.ones(true_count)

            richness = float(np.mean(richness_values)) if len(richness_values) > 0 else 0.0

            metrics_per_task[task] = {
                "scale": scale,
                "coverage": coverage,
                "authenticity": authenticity,
                "richness": richness
            }

        return metrics_per_task
    
    def count_h(self, s, c, a, r):
        r, delta, err_gen, err_emp = s, 1-c, 1-a, 1-r
        h = delta * np.exp(2 * r * (err_gen - err_emp) ** 2) 

        return h
    
    def fit_lower_exp_function(self, x_list, y_list, lambda_range=(1e-4, 10.0), n_search=1000):
        cleaned = [
            (x, y) for x, y in zip(x_list, y_list)
            if x is not None and y is not None
            and isinstance(x, (int, float)) and isinstance(y, (int, float))
            and not (np.isnan(x) or np.isnan(y))
            and x > 0 and y > 0
        ]

        if len(cleaned) == 0:
            raise ValueError("无有效的 (x, y) 数据可用于拟合。")

        x, y = zip(*cleaned)
        x = np.array(x)
        y = np.array(y)


        if np.any((x <= 0) | (y <= 0)):
            raise ValueError("x 和 y 必须为正值。")

        best_a = 0
        best_lambda = None

        lambda_vals = np.logspace(np.log10(lambda_range[0]), np.log10(lambda_range[1]), n_search)

        for lam in lambda_vals:
            f = 1 - np.exp(-lam * x)
            if np.any(f <= 0):
                continue

            a_max = np.min(y / f)        

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
        epsilon = 0.01      # general - empirical
        # epsilon = 1 - (0.99) ** (1/class_num)

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

            diff = np.exp(logterm1 - M) - np.exp(logterm2 - M)
            return np.exp(M) * diff - delta_E

        x_min = np.log(np.max(H_list)) / 2 + 1e-3  
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
        
    def predict_foundation_total_general(self, H_list, delta_E=0.01, epsilon=0.01, verbose=False, max_order=None):
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
    
    def predict_foundation_total_fast(self, H_list, ratio, delta_E=0.01, epsilon=0.01, verbose=False, max_order=5, sample_per_order=200):
        H_list = np.array(H_list)
        k = len(H_list)
        if np.any(H_list <= 0):
            raise ValueError("All h_j must be > 0")

        log_h_list = np.log(H_list)
        log_S_terms = []
        for r in range(1, max_order + 1):
            combs = list(combinations(range(k), r))
            if len(combs) > sample_per_order:
                combs = random.sample(combs, sample_per_order)
            log_products = [np.sum(log_h_list[list(idxs)]) for idxs in combs]
            log_S_r = logsumexp(log_products) if log_products else -np.inf
            log_S_terms.append(log_S_r)

        signs = np.array([(-1)**(r + 1) for r in range(1, max_order + 1)])

        def f(x):
            log_terms = np.array([-2 * (r + 1) * x + log_S_terms[r] for r in range(max_order)])
            max_log_term = np.max(log_terms)
            signed_sum = np.sum(signs * np.exp(log_terms - max_log_term))
            return np.exp(max_log_term) * signed_sum - delta_E

        x_min = max(np.log(np.max(H_list)) / 2, 1e-3)
        x_max = x_min + 30

        try:
            x_solution = brentq(f, x_min, x_max)
            n = (np.log(ratio) * x_solution) / epsilon**2         # 修改了foundation size 的尺寸。
            if verbose:
                print(f"Solution x = {x_solution}, n = {n}")
            return n
        except Exception as e:
            if verbose:
                print("❌ Solve failed:", e)
            return None
        
    
    def predict_foundation_total_simple(self, H_list, delta_E=0.01, epsilon=0.01):
        H_list = np.array(H_list)
        
        if np.any(H_list <= 0):
            raise ValueError("All h_j must be > 0")
        

        # log_delta_sumh = - np.log(delta_E / np.sum(H_list))
        log_delta_sumh = np.log(len(H_list)) * (np.log(np.sum(H_list)) - np.log(delta_E))

        return log_delta_sumh / epsilon**2
    

if __name__ == "__main__":
    calculation = SCARcalculation()
    with open("src/scar/hs_list_cifar10_dino_primary.json", "r") as f:
        data = [json.loads(line) for line in f if line.strip()]

    ratio = data[0]
    hs_list = data[1]

    result = calculation.predict_foundation_total(hs_list, epsilon=0.01)
    print(result)
    
    # result = calculation.predict_foundation_total_general(hs_list, epsilon=0.01)
    # result = calculation.predict_foundation_total_simple(hs_list, epsilon=0.01)
    result = calculation.predict_foundation_total_fast(hs_list, ratio, max_order=3)
    print(result)
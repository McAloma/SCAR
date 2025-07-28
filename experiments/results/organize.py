import re
from collections import defaultdict

def extract_metrics_from_txt_v2(path):
    with open(path, 'r') as f:
        content = f.read()

    # 每个大块实验以两行 = 分割（共 7 段 + 空行 + =，每个完整结构约 8 段）
    blocks = [b.strip() for b in content.split("======================================================================") if b.strip()]

    encoder_count = defaultdict(int)
    result_dict = {}

    # 每 6 个连续段落构成一个实验组（因为你有 6 段内容）
    for i in range(0, len(blocks), 6):
        try:
            title = blocks[i]
            data_line = blocks[i + 1]
            foundation_line = blocks[i + 2]
            test_acc_block = blocks[i + 3]
            extra_acc_block = blocks[i + 4]
            # SCAR block 在 blocks[i + 5]，不处理但占位

            # --- 获取 encoder 名 ---
            encoder_match = re.search(r"Encoder:\s*(\w+)", title)
            if not encoder_match:
                continue
            encoder = encoder_match.group(1)
            encoder_idx = f"{encoder}_{encoder_count[encoder]}"
            encoder_count[encoder] += 1

            # --- 提取字段 ---
            def extract(pattern, text, default="nan", to_type=float):
                match = re.search(pattern, text)
                return to_type(match.group(1)) if match else to_type(default)

            total_acc = extract(r"Total Data Test Acc.*?runs:\s*([\d\.]+)", test_acc_block)
            extend_size = extract(r"Extend datasize:\s*(\d+)", data_line, default="-1", to_type=int)
            total_foundation = extract(r"Total Foundation data size estimation:\s*([\d\.]+)", foundation_line)
            primary_acc = extract(r"Primary Data Test Acc.*?runs:\s*([\d\.]+)", test_acc_block)
            extend_acc = extract(r"Extend Data Test Acc.*?runs:\s*([\d\.]+)", test_acc_block)
            random_acc = extract(r"Random Extend Data Test Acc.*?runs:\s*([\d\.]+)", extra_acc_block)
            avg_acc = extract(r"Average Extend Data Test Acc.*?runs:\s*([\d\.]+)", extra_acc_block)

            result_dict[encoder_idx] = [
                total_acc,
                extend_size,
                total_foundation,
                primary_acc,
                extend_acc,
                random_acc,
                avg_acc,
            ]
        except Exception as e:
            print(f"❗️跳过 block @ index {i}，原因：{e}")
            continue

    return result_dict


# 示例使用
if __name__ == "__main__":
    input_file = "experiments/results/wikipedia_experiment_results.txt"  # 修改为你的实际 txt 文件路径
    result = extract_metrics_from_txt_v2(input_file)

    sorted_result = dict(sorted(result.items()))

    print(sorted_result.keys())

    for key, values in sorted_result.items():
        print(f"        {values},")
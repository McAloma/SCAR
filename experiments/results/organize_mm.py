import re
from collections import defaultdict

def extract_metrics_from_txt_v3(path):
    with open(path, 'r') as f:
        content = f.read()

    # 以 "======" 分割，获取每个 block（注意内容较多，要 strip）
    blocks = [b.strip() for b in content.split("=" * 70) if b.strip()]

    encoder_count = defaultdict(int)
    result_dict = {}

    # 每 3 个 block 为一组（title / text / image）
    for i in range(0, len(blocks), 3):
        try:
            title = blocks[i]
            text_block = blocks[i + 1]
            image_block = blocks[i + 2]

            # --- 获取 encoder 名 ---
            encoder_match = re.search(r"Encoder:\s*(\w+)", title)
            if not encoder_match:
                continue
            encoder = encoder_match.group(1)
            encoder_idx = f"{encoder}_{encoder_count[encoder]}"
            encoder_count[encoder] += 1

            # --- 提取函数 ---
            def extract(pattern, text, default="nan", to_type=float):
                match = re.search(pattern, text)
                return to_type(match.group(1)) if match else to_type(default)

            def parse_modality_block(modality_text):
                extend_size = extract(r"Extend datasize:\s*(\d+)", modality_text, default="-1", to_type=int)
                total_foundation = extract(r"Total Foundation data size estimation:\s*([\d\.]+)", modality_text)
                total_acc = extract(r"Total Data Test Acc:\s*([\d\.]+)", modality_text)
                primary_acc = extract(r"Primary Data Test Acc:\s*([\d\.]+)", modality_text)
                extend_acc = extract(r"Extend Data Test Acc:\s*([\d\.]+)", modality_text)
                random_acc = extract(r"Random Extend Data Test Acc:\s*([\d\.]+)", modality_text)
                avg_acc = extract(r"Average Extend Data Test Acc:\s*([\d\.]+)", modality_text)

                return [
                    total_acc,
                    extend_size,
                    total_foundation,
                    primary_acc,
                    extend_acc,
                    random_acc,
                    avg_acc,
                ]

            # 存两个 modality 的指标
            result_dict[f"text_{encoder_idx}"] = parse_modality_block(text_block)
            result_dict[f"image_{encoder_idx}"] = parse_modality_block(image_block)

        except Exception as e:
            print(f"❗️跳过 block @ index {i}，原因：{e}")
            continue

    return result_dict


# 示例使用
if __name__ == "__main__":
    # input_file = "experiments/results/msrvtt_experiment_results.txt"  # 修改为你的实际路径
    input_file = "test_hanlin/results/hanlin_result.txt"
    result = extract_metrics_from_txt_v3(input_file)

    sorted_result = dict(sorted(result.items()))

    print(sorted_result.keys())

    for key, values in sorted_result.items():
        print(f"        {values},")
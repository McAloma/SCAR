import re
import numpy as np
from collections import defaultdict

def parse_experiment_log_from_string(log_text: str):
    # 初始化结构
    ratios = []
    metrics = {
        'WHAT': defaultdict(list),
        'WHERE': defaultdict(list),
        'HOW': defaultdict(list)
    }

    # 分割每次实验
    experiments = re.split(r"=+\n", log_text)
    for exp in experiments:
        # 提取 Ratio，使用括号右边的小数
        ratio_match = re.search(r"\[Ratio\]\s*[^\n]*=\s*([0-9.]+)", exp)
        if not ratio_match:
            continue
        ratio = float(ratio_match.group(1))
        ratios.append(ratio)

        # 提取每一类 WHAT / WHERE / HOW 的 SCAR 指标
        for section in ['WHAT', 'WHERE', 'HOW']:
            for metric in ['scale', 'coverage', 'authenticity', 'richness']:
                pattern = rf"{section}:\s*.*?{metric}\s*:\s*([0-9.]+)"
                match = re.search(pattern, exp, re.DOTALL)
                if match:
                    metrics[section][metric].append(float(match.group(1)))
                else:
                    metrics[section][metric].append(np.nan)

    # 按比例升序排序
    sorted_indices = np.argsort(ratios)
    ratios_sorted = np.array(ratios)[sorted_indices]

    print("x = np.array(", ratios_sorted.tolist(), ")\n")

    for section in ['WHAT', 'WHERE', 'HOW']:
        print(f"# {section}")
        for metric in ['scale', 'coverage', 'authenticity', 'richness']:
            values = np.array(metrics[section][metric])[sorted_indices]
            print(f"{metric}_{section.lower()} = np.array({values.tolist()})")
        print()

# ✅ 用法示例：你只需要把 log_text 替换成你自己的记录字符串
if __name__ == "__main__":
    print("# ResNet Log\n")
    log_text = """
    ============================================================
[Timestamp]      2025-05-19 18:51:26
[Encoder]        resnet
[Ratio]          1/1 = 1.00

[False Sample Statistics]
  - Avg False Sample per Fold     : 1786.6667
  - Avg False Sample (Total)      : 151.0000

[Average SCAR Index per Fold]
  - WHAT:
      scale               : 0.9970
      coverage            : 0.9994
      authenticity        : 0.9874
      richness            : 0.9985
  - WHERE:
      scale               : 0.9970
      coverage            : 0.9999
      authenticity        : 0.9472
      richness            : 0.9852
  - HOW:
      scale               : 0.9970
      coverage            : 0.9994
      authenticity        : 0.9582
      richness            : 0.9842

[Average K-Fold Accuracies]
  - WHAT                           : 0.9874
  - WHERE                          : 0.9472
  - HOW                            : 0.9582

[Average Test Accuracies]
  - WHAT                           : 0.9115
  - WHERE                          : 0.9108
  - HOW                            : 0.9116
============================================================

============================================================
[Timestamp]      2025-05-19 18:57:58
[Encoder]        resnet
[Ratio]          1/2 = 0.50

[False Sample Statistics]
  - Avg False Sample per Fold     : 1791.3333
  - Avg False Sample (Total)      : 10.0000

[Average SCAR Index per Fold]
  - WHAT:
      scale               : 0.4998
      coverage            : 0.9995
      authenticity        : 0.9735
      richness            : 0.9974
  - WHERE:
      scale               : 0.4998
      coverage            : 0.9999
      authenticity        : 0.9117
      richness            : 0.9937
  - HOW:
      scale               : 0.4998
      coverage            : 0.9993
      authenticity        : 0.8998
      richness            : 0.9935

[Average K-Fold Accuracies]
  - WHAT                           : 0.9735
  - WHERE                          : 0.9117
  - HOW                            : 0.8998

[Average Test Accuracies]
  - WHAT                           : 0.8993
  - WHERE                          : 0.9009
  - HOW                            : 0.8989
============================================================

============================================================
[Timestamp]      2025-05-19 19:09:17
[Encoder]        resnet
[Ratio]          1/5 = 0.20

[False Sample Statistics]
  - Avg False Sample per Fold     : 806.8000
  - Avg False Sample (Total)      : 7.2000

[Average SCAR Index per Fold]
  - WHAT:
      scale               : 0.1999
      coverage            : 0.9991
      authenticity        : 0.9685
      richness            : 0.9964
  - WHERE:
      scale               : 0.1999
      coverage            : 0.9993
      authenticity        : 0.8965
      richness            : 0.9908
  - HOW:
      scale               : 0.1999
      coverage            : 0.9977
      authenticity        : 0.8930
      richness            : 0.9908

[Average K-Fold Accuracies]
  - WHAT                           : 0.9685
  - WHERE                          : 0.8965
  - HOW                            : 0.8930

[Average Test Accuracies]
  - WHAT                           : 0.8831
  - WHERE                          : 0.8819
  - HOW                            : 0.8833
============================================================

============================================================
[Timestamp]      2025-05-19 19:28:30
[Encoder]        resnet
[Ratio]          1/10 = 0.10

[False Sample Statistics]
  - Avg False Sample per Fold     : 449.6000
  - Avg False Sample (Total)      : 6.0000

[Average SCAR Index per Fold]
  - WHAT:
      scale               : 0.0999
      coverage            : 0.9956
      authenticity        : 0.9638
      richness            : 0.9949
  - WHERE:
      scale               : 0.0999
      coverage            : 0.9978
      authenticity        : 0.8788
      richness            : 0.9874
  - HOW:
      scale               : 0.0999
      coverage            : 0.9943
      authenticity        : 0.8876
      richness            : 0.9875

[Average K-Fold Accuracies]
  - WHAT                           : 0.9638
  - WHERE                          : 0.8788
  - HOW                            : 0.8876

[Average Test Accuracies]
  - WHAT                           : 0.8664
  - WHERE                          : 0.8659
  - HOW                            : 0.8671
============================================================

============================================================
[Timestamp]      2025-05-19 20:03:01
[Encoder]        resnet
[Ratio]          1/20 = 0.05

[False Sample Statistics]
  - Avg False Sample per Fold     : 264.3500
  - Avg False Sample (Total)      : 6.3500

[Average SCAR Index per Fold]
  - WHAT:
      scale               : 0.0499
      coverage            : 0.9921
      authenticity        : 0.9578
      richness            : 0.9936
  - WHERE:
      scale               : 0.0499
      coverage            : 0.9959
      authenticity        : 0.8482
      richness            : 0.9798
  - HOW:
      scale               : 0.0499
      coverage            : 0.9918
      authenticity        : 0.8768
      richness            : 0.9807

[Average K-Fold Accuracies]
  - WHAT                           : 0.9578
  - WHERE                          : 0.8482
  - HOW                            : 0.8768

[Average Test Accuracies]
  - WHAT                           : 0.8478
  - WHERE                          : 0.8473
  - HOW                            : 0.8470
============================================================

============================================================
[Timestamp]      2025-05-19 21:24:50
[Encoder]        resnet
[Ratio]          1/50 = 0.02

[False Sample Statistics]
  - Avg False Sample per Fold     : 150.0133
  - Avg False Sample (Total)      : 6.1000

[Average SCAR Index per Fold]
  - WHAT:
      scale               : 0.0199
      coverage            : 0.9875
      authenticity        : 0.9523
      richness            : 0.9913
  - WHERE:
      scale               : 0.0199
      coverage            : 0.9855
      authenticity        : 0.7602
      richness            : 0.9536
  - HOW:
      scale               : 0.0199
      coverage            : 0.9710
      authenticity        : 0.8374
      richness            : 0.9555

[Average K-Fold Accuracies]
  - WHAT                           : 0.9523
  - WHERE                          : 0.7602
  - HOW                            : 0.8374

[Average Test Accuracies]
  - WHAT                           : 0.8209
  - WHERE                          : 0.8216
  - HOW                            : 0.8205
============================================================

============================================================
Experiment completed at 2025-05-19 21:24:50
============================================================
    """
    parse_experiment_log_from_string(log_text)


    print("# ViT Log\n")
    log_text = """
    ============================================================
[Timestamp]      2025-05-19 21:30:42
[Encoder]        vit
[Ratio]          1/1 = 1.00

[False Sample Statistics]
  - Avg False Sample per Fold     : 572.6667
  - Avg False Sample (Total)      : 73.0000

[Average SCAR Index per Fold]
  - WHAT:
      scale               : 0.9985
      coverage            : 0.9978
      authenticity        : 0.9949
      richness            : 0.9993
  - WHERE:
      scale               : 0.9985
      coverage            : 0.9957
      authenticity        : 0.9840
      richness            : 0.9955
  - HOW:
      scale               : 0.9985
      coverage            : 0.9774
      authenticity        : 0.9868
      richness            : 0.9950

[Average K-Fold Accuracies]
  - WHAT                           : 0.9949
  - WHERE                          : 0.9840
  - HOW                            : 0.9868

[Average Test Accuracies]
  - WHAT                           : 0.9653
  - WHERE                          : 0.9655
  - HOW                            : 0.9657
============================================================

============================================================
[Timestamp]      2025-05-19 21:35:27
[Encoder]        vit
[Ratio]          1/2 = 0.50

[False Sample Statistics]
  - Avg False Sample per Fold     : 852.3333
  - Avg False Sample (Total)      : 1.0000

[Average SCAR Index per Fold]
  - WHAT:
      scale               : 0.5000
      coverage            : 0.9991
      authenticity        : 0.9904
      richness            : 0.9996
  - WHERE:
      scale               : 0.5000
      coverage            : 0.9951
      authenticity        : 0.9769
      richness            : 0.9992
  - HOW:
      scale               : 0.5000
      coverage            : 0.9869
      authenticity        : 0.9305
      richness            : 0.9990

[Average K-Fold Accuracies]
  - WHAT                           : 0.9904
  - WHERE                          : 0.9769
  - HOW                            : 0.9305

[Average Test Accuracies]
  - WHAT                           : 0.9637
  - WHERE                          : 0.9628
  - HOW                            : 0.9633
============================================================

============================================================
[Timestamp]      2025-05-19 21:42:50
[Encoder]        vit
[Ratio]          1/5 = 0.20

[False Sample Statistics]
  - Avg False Sample per Fold     : 343.2000
  - Avg False Sample (Total)      : 0.6000

[Average SCAR Index per Fold]
  - WHAT:
      scale               : 0.2000
      coverage            : 0.9957
      authenticity        : 0.9882
      richness            : 0.9994
  - WHERE:
      scale               : 0.2000
      coverage            : 0.9901
      authenticity        : 0.9733
      richness            : 0.9988
  - HOW:
      scale               : 0.2000
      coverage            : 0.9678
      authenticity        : 0.9355
      richness            : 0.9988

[Average K-Fold Accuracies]
  - WHAT                           : 0.9882
  - WHERE                          : 0.9733
  - HOW                            : 0.9355

[Average Test Accuracies]
  - WHAT                           : 0.9602
  - WHERE                          : 0.9606
  - HOW                            : 0.9609
============================================================

============================================================
[Timestamp]      2025-05-19 21:54:40
[Encoder]        vit
[Ratio]          1/10 = 0.10

[False Sample Statistics]
  - Avg False Sample per Fold     : 168.1000
  - Avg False Sample (Total)      : 0.1000

[Average SCAR Index per Fold]
  - WHAT:
      scale               : 0.1000
      coverage            : 0.9959
      authenticity        : 0.9865
      richness            : 0.9995
  - WHERE:
      scale               : 0.1000
      coverage            : 0.9846
      authenticity        : 0.9694
      richness            : 0.9989
  - HOW:
      scale               : 0.1000
      coverage            : 0.9524
      authenticity        : 0.9432
      richness            : 0.9987

[Average K-Fold Accuracies]
  - WHAT                           : 0.9865
  - WHERE                          : 0.9694
  - HOW                            : 0.9432

[Average Test Accuracies]
  - WHAT                           : 0.9587
  - WHERE                          : 0.9586
  - HOW                            : 0.9586
============================================================

============================================================
[Timestamp]      2025-05-19 22:14:22
[Encoder]        vit
[Ratio]          1/20 = 0.05

[False Sample Statistics]
  - Avg False Sample per Fold     : 89.9167
  - Avg False Sample (Total)      : 0.0500

[Average SCAR Index per Fold]
  - WHAT:
      scale               : 0.0500
      coverage            : 0.9938
      authenticity        : 0.9806
      richness            : 0.9990
  - WHERE:
      scale               : 0.0500
      coverage            : 0.9726
      authenticity        : 0.9574
      richness            : 0.9977
  - HOW:
      scale               : 0.0500
      coverage            : 0.9416
      authenticity        : 0.9542
      richness            : 0.9978

[Average K-Fold Accuracies]
  - WHAT                           : 0.9806
  - WHERE                          : 0.9574
  - HOW                            : 0.9542

[Average Test Accuracies]
  - WHAT                           : 0.9559
  - WHERE                          : 0.9565
  - HOW                            : 0.9561
============================================================

============================================================
[Timestamp]      2025-05-19 22:59:27
[Encoder]        vit
[Ratio]          1/50 = 0.02

[False Sample Statistics]
  - Avg False Sample per Fold     : 83.9133
  - Avg False Sample (Total)      : 1.2800

[Average SCAR Index per Fold]
  - WHAT:
      scale               : 0.0200
      coverage            : 0.9920
      authenticity        : 0.9688
      richness            : 0.9958
  - WHERE:
      scale               : 0.0200
      coverage            : 0.9520
      authenticity        : 0.8646
      richness            : 0.9776
  - HOW:
      scale               : 0.0200
      coverage            : 0.9120
      authenticity        : 0.9149
      richness            : 0.9798

[Average K-Fold Accuracies]
  - WHAT                           : 0.9688
  - WHERE                          : 0.8646
  - HOW                            : 0.9149

[Average Test Accuracies]
  - WHAT                           : 0.9497
  - WHERE                          : 0.9496
  - HOW                            : 0.9499
============================================================

============================================================
Experiment completed at 2025-05-19 22:59:27
============================================================
    """
    parse_experiment_log_from_string(log_text)


    print("# DINO Log\n")
    log_text = """
    ============================================================
[Timestamp]      2025-05-19 23:05:21
[Encoder]        dino
[Ratio]          1/1 = 1.00

[False Sample Statistics]
  - Avg False Sample per Fold     : 236.6667
  - Avg False Sample (Total)      : 24.0000

[Average SCAR Index per Fold]
  - WHAT:
      scale               : 0.9995
      coverage            : 0.9983
      authenticity        : 0.9980
      richness            : 0.9998
  - WHERE:
      scale               : 0.9995
      coverage            : 0.9950
      authenticity        : 0.9935
      richness            : 0.9982
  - HOW:
      scale               : 0.9995
      coverage            : 0.9582
      authenticity        : 0.9943
      richness            : 0.9980

[Average K-Fold Accuracies]
  - WHAT                           : 0.9980
  - WHERE                          : 0.9935
  - HOW                            : 0.9943

[Average Test Accuracies]
  - WHAT                           : 0.9839
  - WHERE                          : 0.9832
  - HOW                            : 0.9842
============================================================

============================================================
[Timestamp]      2025-05-19 23:10:08
[Encoder]        dino
[Ratio]          1/2 = 0.50

[False Sample Statistics]
  - Avg False Sample per Fold     : 1063.1667
  - Avg False Sample (Total)      : 1.0000

[Average SCAR Index per Fold]
  - WHAT:
      scale               : 0.5000
      coverage            : 0.9998
      authenticity        : 0.9919
      richness            : 0.9994
  - WHERE:
      scale               : 0.5000
      coverage            : 0.9933
      authenticity        : 0.9735
      richness            : 0.9987
  - HOW:
      scale               : 0.5000
      coverage            : 0.9700
      authenticity        : 0.9070
      richness            : 0.9983

[Average K-Fold Accuracies]
  - WHAT                           : 0.9919
  - WHERE                          : 0.9735
  - HOW                            : 0.9070

[Average Test Accuracies]
  - WHAT                           : 0.9829
  - WHERE                          : 0.9840
  - HOW                            : 0.9832
============================================================

============================================================
[Timestamp]      2025-05-19 23:17:30
[Encoder]        dino
[Ratio]          1/5 = 0.20

[False Sample Statistics]
  - Avg False Sample per Fold     : 439.1333
  - Avg False Sample (Total)      : 0.4000

[Average SCAR Index per Fold]
  - WHAT:
      scale               : 0.2000
      coverage            : 0.9982
      authenticity        : 0.9866
      richness            : 0.9986
  - WHERE:
      scale               : 0.2000
      coverage            : 0.9909
      authenticity        : 0.9702
      richness            : 0.9987
  - HOW:
      scale               : 0.2000
      coverage            : 0.9514
      authenticity        : 0.9114
      richness            : 0.9979

[Average K-Fold Accuracies]
  - WHAT                           : 0.9866
  - WHERE                          : 0.9702
  - HOW                            : 0.9114

[Average Test Accuracies]
  - WHAT                           : 0.9817
  - WHERE                          : 0.9816
  - HOW                            : 0.9815
============================================================

============================================================
[Timestamp]      2025-05-19 23:29:16
[Encoder]        dino
[Ratio]          1/10 = 0.10

[False Sample Statistics]
  - Avg False Sample per Fold     : 252.1667
  - Avg False Sample (Total)      : 0.6000

[Average SCAR Index per Fold]
  - WHAT:
      scale               : 0.1000
      coverage            : 0.9966
      authenticity        : 0.9851
      richness            : 0.9986
  - WHERE:
      scale               : 0.1000
      coverage            : 0.9848
      authenticity        : 0.9669
      richness            : 0.9984
  - HOW:
      scale               : 0.1000
      coverage            : 0.9468
      authenticity        : 0.8968
      richness            : 0.9975

[Average K-Fold Accuracies]
  - WHAT                           : 0.9851
  - WHERE                          : 0.9669
  - HOW                            : 0.8968

[Average Test Accuracies]
  - WHAT                           : 0.9802
  - WHERE                          : 0.9798
  - HOW                            : 0.9799
============================================================

============================================================
[Timestamp]      2025-05-19 23:48:56
[Encoder]        dino
[Ratio]          1/20 = 0.05

[False Sample Statistics]
  - Avg False Sample per Fold     : 151.6500
  - Avg False Sample (Total)      : 0.5500

[Average SCAR Index per Fold]
  - WHAT:
      scale               : 0.0500
      coverage            : 0.9947
      authenticity        : 0.9803
      richness            : 0.9980
  - WHERE:
      scale               : 0.0500
      coverage            : 0.9833
      authenticity        : 0.9561
      richness            : 0.9976
  - HOW:
      scale               : 0.0500
      coverage            : 0.9375
      authenticity        : 0.8816
      richness            : 0.9965

[Average K-Fold Accuracies]
  - WHAT                           : 0.9803
  - WHERE                          : 0.9561
  - HOW                            : 0.8816

[Average Test Accuracies]
  - WHAT                           : 0.9769
  - WHERE                          : 0.9772
  - HOW                            : 0.9774
============================================================

============================================================
[Timestamp]      2025-05-20 00:33:48
[Encoder]        dino
[Ratio]          1/50 = 0.02

[False Sample Statistics]
  - Avg False Sample per Fold     : 68.6467
  - Avg False Sample (Total)      : 0.3800

[Average SCAR Index per Fold]
  - WHAT:
      scale               : 0.0200
      coverage            : 0.9925
      authenticity        : 0.9710
      richness            : 0.9972
  - WHERE:
      scale               : 0.0200
      coverage            : 0.9707
      authenticity        : 0.9439
      richness            : 0.9964
  - HOW:
      scale               : 0.0200
      coverage            : 0.9168
      authenticity        : 0.8791
      richness            : 0.9955

[Average K-Fold Accuracies]
  - WHAT                           : 0.9710
  - WHERE                          : 0.9439
  - HOW                            : 0.8791

[Average Test Accuracies]
  - WHAT                           : 0.9709
  - WHERE                          : 0.9711
  - HOW                            : 0.9710
============================================================

============================================================
Experiment completed at 2025-05-20 00:33:48
============================================================
    """
    parse_experiment_log_from_string(log_text)
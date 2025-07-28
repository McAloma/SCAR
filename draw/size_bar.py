import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import matplotlib.ticker as ticker

# ==== 1. 定义数据 ====
size_records = [
    ('CIFAR-10', 'ResNet50', 89.92, 48.97, 50),
    ('CIFAR-10', 'ViT-B/16', 96.58, 48.12, 50),
    ('CIFAR-10', 'DINO-v2', 93.40, 49.43, 50),

    ('CIFAR-100', 'ResNet50', 294.47, 48.68, 50),
    ('CIFAR-100', 'ViT-B/16', 293.78, 49.15, 50),
    ('CIFAR-100', 'DINO-v2', 294.51, 49.34, 50),

    ('ImageNet-1K', 'ResNet50', 528.86, 1027.60, 1281),
    ('ImageNet-1K', 'ViT-B/16', 476.97, 1045.62, 1281),
    ('ImageNet-1K', 'DINO-v2', 472.78, 1043.98, 1281),

    ('AG-News', 'BERT', 65.92, 119.48, 120),
    ('AG-News', 'RoBERTa', 57.41, 116.56, 120),
    ('AG-News', 'GPT2', 71.24, 118.51, 120),

    ('DBPedia', 'BERT', 137.47, 475.12, 560),
    ('DBPedia', 'RoBERTa', 158.49, 462.81, 560),
    ('DBPedia', 'GPT2', 161.15, 466.78, 560),

    ('Wikipedia', 'BERT', 351.09, 584.63, 945),
    ('Wikipedia', 'RoBERTa', 360.06, 588.38, 945),
    ('Wikipedia', 'GPT2', 356.60, 588.43, 947),

    ('Flickr-Text', 'CLIP', 121, 20, 24),
    ('Flickr-Text', 'CoCa', 117, 20, 24),
    ('Flickr-Text', 'SigLIP', 126, 20, 24),

    ('Flickr-Image', 'CLIP', 113, 19, 24),
    ('Flickr-Image', 'CoCa', 119, 18, 24),
    ('Flickr-Image', 'SigLIP', 128, 16, 24),

    ('COCO-Text', 'CLIP', 257, 58, 82),
    ('COCO-Text', 'CoCa', 256, 60, 82),
    ('COCO-Text', 'SigLIP', 293, 63, 82),

    ('COCO-Image', 'CLIP', 254, 58, 82),
    ('COCO-Image', 'CoCa', 260, 58, 82),
    ('COCO-Image', 'SigLIP', 283, 53, 82),

    ('MSR-VTT-Text', 'X-CLIP', 73, 7.13, 9),
    ('MSR-VTT-Video', 'X-CLIP', 64.18, 7.77, 9),

    ('AudioCaps-Text', 'CLAP', 50.89, 40.53, 45),
    ('AudioCaps-Audio', 'CLAP', 55.56, 40.40, 45),
]

df = pd.DataFrame(size_records, columns=['Dataset', 'Model', 'Fou_size_k', 'Ext_size_k', 'Total_size_k'])

# ==== 2. 按 Dataset 分组，取每个数据集下多个模型的平均 ====
df_avg = df.groupby('Dataset')[['Fou_size_k', 'Ext_size_k', 'Total_size_k']].mean().reset_index()

# ==== 3. 画图 ====
plt.figure(figsize=(11, 7))

x = np.arange(len(df_avg))
width = 0.35

bars_total = plt.bar(x - width/2, df_avg['Total_size_k'], width, label='Total Size (k)', color='lightgreen', edgecolor='black', zorder=3)
bars_ext = plt.bar(x - width/2, df_avg['Ext_size_k'], width, label='Extension Size (k)', color='salmon', alpha=0.7, edgecolor='black', zorder=4)
bars_fou = plt.bar(x + width/2, df_avg['Fou_size_k'], width, label='Foundation Size (k)', color='skyblue', edgecolor='black', zorder=3)

# ==== 设置对数轴 ====
plt.yscale('log', base=2)
plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=2.0, numticks=15))
plt.gca().yaxis.set_minor_locator(ticker.LogLocator(base=2.0, subs='auto', numticks=100))
plt.gca().yaxis.set_minor_formatter(ticker.NullFormatter())

# plt.xticks(x + width / 2, df_avg['Dataset'], rotation=45, ha='right', fontsize=9)
plt.xticks(x, df_avg['Dataset'],rotation=45, fontsize=9)
plt.ylabel('Size (thousands)', fontsize=12)
# plt.title('Average Total, Extension, and Foundation Size per Dataset', fontsize=14)
plt.legend()

# 数字标签
def add_labels_total(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{int(height)}',
            ha='center', va='bottom',
            fontsize=7,
            fontweight='bold'
        )

def add_labels_ext(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height * 0.88,  # 数字放在柱底部内侧，0.05 倍柱高位置
            f'{int(height)}',
            ha='center', va='bottom',
            fontsize=7,
            fontweight='bold',
        )

def add_labels_fou(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{int(height)}',
            ha='center', va='bottom',
            fontsize=7,
            fontweight='bold'
        )

add_labels_total(bars_total)
add_labels_ext(bars_ext)
add_labels_fou(bars_fou)

plt.tight_layout()
os.makedirs('draw/pics', exist_ok=True)
plt.savefig('draw/pics/data_size_barplot_avg.png', dpi=300)
plt.close()
print("✅ 平均柱状图已保存到 draw/pics/data_size_barplot.png")
# import matplotlib.pyplot as plt
# import pandas as pd
# import os

# # ==== 1. 定义数据 ====
# scar_records = [
#     ('CIFAR-10', 'ResNet50', 0.5560, 0.9484, 0.9154, 1.0000),
#     ('CIFAR-10', 'ViT-B/16', 0.5177, 0.9060, 0.9627, 1.0000),
#     ('CIFAR-10', 'DINO-v2', 0.5353, 0.9717, 0.9818, 1.0000),

#     ('CIFAR-100', 'ResNet50', 0.1698, 0.9339, 0.9424, 1.0000),
#     ('CIFAR-100', 'ViT-B/16', 0.1702, 0.9575, 0.9540, 1.0000),
#     ('CIFAR-100', 'DINO-v2', 0.1698, 0.9672, 0.9809, 1.0000),

#     ('ImageNet-1K', 'ResNet50', 1.0000, 0.5052, 0.7561, 0.4128),
#     ('ImageNet-1K', 'ViT-B/16', 1.0000, 0.5404, 0.7701, 0.3723),
#     ('ImageNet-1K', 'DINO-v2', 1.0000, 0.5372, 0.7747, 0.3690),

#     ('AG-News', 'BERT', 1.0000, 0.9891, 0.9023, 0.5493),
#     ('AG-News', 'RoBERTa', 1.0000, 0.9284, 0.9053, 0.4784),
#     ('AG-News', 'GPT2', 1.0000, 0.9689, 0.9147, 0.5937),

#     ('DBPedia', 'BERT', 1.0000, 0.6211, 0.9898, 0.2455),
#     ('DBPedia', 'RoBERTa', 1.0000, 0.5661, 0.9859, 0.2830),
#     ('DBPedia', 'GPT2', 1.0000, 0.5839, 0.9839, 0.2878),

#     ('Wikipedia', 'BERT', 1.0000, 0.0462, 0.4051, 0.3714),
#     ('Wikipedia', 'RoBERTa', 1.0000, 0.0562, 0.1637, 0.3809),
#     ('Wikipedia', 'GPT2', 1.0000, 0.0532, 0.3846, 0.3773),

#     ('Flickr30k-Text', 'CLIP', 0.2041, 0.5368, 0.3214, 1.0000),
#     ('Flickr30k-Text', 'CoCa', 0.2109, 0.5388, 0.4680, 1.0000),
#     ('Flickr30k-Text', 'SigLIP', 0.1967, 0.5911, 0.2054, 1.0000),

#     ('Flickr30k-Image', 'CLIP', 0.2187, 0.5120, 0.3889, 1.0000),
#     ('Flickr30k-Image', 'CoCa', 0.2070, 0.4083, 0.5033, 1.0000),
#     ('Flickr30k-Image', 'SigLIP', 0.1932, 0.1298, 0.2562, 1.0000),

#     ('COCOCap-Text', 'CLIP', 0.3219, 0.2632, 0.5930, 1.0000),
#     ('COCOCap-Text', 'CoCa', 0.3226, 0.3263, 0.6002, 1.0000),
#     ('COCOCap-Text', 'SigLIP', 0.2818, 0.4251, 0.1431, 1.0000),

#     ('COCOCap-Image', 'CLIP', 0.3249, 0.2573, 0.6228, 1.0000),
#     ('COCOCap-Image', 'CoCa', 0.3176, 0.2610, 0.6100, 1.0000),
#     ('COCOCap-Image', 'SigLIP', 0.2916, 0.1048, 0.2101, 1.0000),
# ]

# df = pd.DataFrame(scar_records, columns=['Dataset', 'Model', 'Scale', 'Coverage', 'Richness', 'Authenticity'])
# df['Label'] = df['Dataset'] + ' / ' + df['Model']

# # ==== 2. 创建保存路径 ====
# os.makedirs("draw/pics", exist_ok=True)

# # ==== 3. 绘图函数 ====
# def plot_modified_bubble(df_all, title, save_path):
#     plt.figure(figsize=(8, 6))
#     scatter = plt.scatter(
#         df_all['Coverage'],
#         df_all['Richness'],
#         s=df_all['Scale'] * 3000,
#         c=df_all['Authenticity'],
#         cmap='Blues',
#         alpha=0.8,
#         edgecolors='black'
#     )

#     for i in range(len(df_all)):
#         x = df_all['Coverage'].iloc[i]
#         y = df_all['Richness'].iloc[i]
#         dataset = df_all['Dataset'].iloc[i]
#         model = df_all['Model'].iloc[i]
#         plt.text(
#             x, y,
#             f"{dataset}\n{model}",
#             fontsize=7.5,
#             ha='center',
#             va='center'
#         )

#     plt.xlabel('Coverage', fontsize=12)
#     plt.ylabel('Richness', fontsize=12)
#     plt.title(title, fontsize=14)
#     plt.grid(True)

#     plt.xlim(-0.2, 1.2)
#     plt.ylim(-0.2, 1.2)

#     cbar = plt.colorbar(scatter)
#     cbar.set_label('Authenticity', fontsize=12)

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.close()
#     print(f"✅ 图保存成功：{save_path}")

# # ==== 4. 调用函数 ====
# plot_modified_bubble(df, "SCAR Bubble (Auth vs. Richness, Color=Coverage)", "draw/pics/SCAR_bubble_all_modified.png")




import matplotlib.pyplot as plt
import pandas as pd
import os

# ==== 1. 定义数据 ====
scar_records = [
    ('CIFAR-10', 'ResNet50', 0.5560, 0.9484, 0.9154, 1.0000),
    ('CIFAR-10', 'ViT-B/16', 0.5177, 0.9060, 0.9627, 1.0000),
    ('CIFAR-10', 'DINO-v2', 0.5353, 0.9717, 0.9818, 1.0000),

    ('CIFAR-100', 'ResNet50', 0.1698, 0.9339, 0.9424, 1.0000),
    ('CIFAR-100', 'ViT-B/16', 0.1702, 0.9575, 0.9540, 1.0000),
    ('CIFAR-100', 'DINO-v2', 0.1698, 0.9672, 0.9809, 1.0000),

    ('ImageNet-1K', 'ResNet50', 1.0000, 0.5052, 0.7561, 0.4128),
    ('ImageNet-1K', 'ViT-B/16', 1.0000, 0.5404, 0.7701, 0.3723),
    ('ImageNet-1K', 'DINO-v2', 1.0000, 0.5372, 0.7747, 0.3690),

    ('AG-News', 'BERT', 1.0000, 0.9891, 0.9023, 0.5493),
    ('AG-News', 'RoBERTa', 1.0000, 0.9284, 0.9053, 0.4784),
    ('AG-News', 'GPT2', 1.0000, 0.9689, 0.9147, 0.5937),

    ('DBPedia', 'BERT', 1.0000, 0.6211, 0.9898, 0.2455),
    ('DBPedia', 'RoBERTa', 1.0000, 0.5661, 0.9859, 0.2830),
    ('DBPedia', 'GPT2', 1.0000, 0.5839, 0.9839, 0.2878),

    ('Wikipedia', 'BERT', 1.0000, 0.0462, 0.4051, 0.3714),
    ('Wikipedia', 'RoBERTa', 1.0000, 0.0562, 0.1637, 0.3809),
    ('Wikipedia', 'GPT2', 1.0000, 0.0532, 0.3846, 0.3773),

    ('Flickr30k-Text', 'CLIP', 0.2041, 0.5368, 0.3214, 1.0000),
    ('Flickr30k-Text', 'CoCa', 0.2109, 0.5388, 0.4680, 1.0000),
    ('Flickr30k-Text', 'SigLIP', 0.1967, 0.5911, 0.2054, 1.0000),

    ('Flickr30k-Image', 'CLIP', 0.2187, 0.5120, 0.3889, 1.0000),
    ('Flickr30k-Image', 'CoCa', 0.2070, 0.4083, 0.5033, 1.0000),
    ('Flickr30k-Image', 'SigLIP', 0.1932, 0.1298, 0.2562, 1.0000),

    ('COCOCap-Text', 'CLIP', 0.3219, 0.2632, 0.5930, 1.0000),
    ('COCOCap-Text', 'CoCa', 0.3226, 0.3263, 0.6002, 1.0000),
    ('COCOCap-Text', 'SigLIP', 0.2818, 0.4251, 0.1431, 1.0000),

    ('COCOCap-Image', 'CLIP', 0.3249, 0.2573, 0.6228, 1.0000),
    ('COCOCap-Image', 'CoCa', 0.3176, 0.2610, 0.6100, 1.0000),
    ('COCOCap-Image', 'SigLIP', 0.2916, 0.1048, 0.2101, 1.0000),
    
    ('MSR-VTT-Text', 'X-CLIP', 0.1229, 0.4816, 0.1733, 1.0000),
    ('MSR-VTT-Video', 'X-CLIP', 0.1402, 0.6592, 0.0617, 1.0000),

    ('AudioCaps', 'CLAP', 0.8877, 0.7428, 0.2666, 1.0000),
    ('AudioCaps', 'CLAP', 0.8131, 0.7356, 0.3482, 1.0000),
]

df = pd.DataFrame(scar_records, columns=['Dataset', 'Model', 'Scale', 'Coverage', 'Richness', 'Authenticity'])

# ==== 2. 缩写函数 ====
def shorten(dataset, model):
    ds_map = {
        'CIFAR-10': 'C10', 'CIFAR-100': 'C100', 'ImageNet-1K': 'IN1K',
        'AG-News': 'AG', 'DBPedia': 'DBP', 'Wikipedia': 'Wiki',
        'Flickr30k-Text': 'F30k-T', 'Flickr30k-Image': 'F30k-I',
        'COCOCap-Text': 'COCO-T', 'COCOCap-Image': 'COCO-I'
    }
    return f"{ds_map.get(dataset, dataset)}-{model}"

df['ShortLabel'] = [shorten(d, m) for d, m in zip(df['Dataset'], df['Model'])]

# ==== 3. 创建保存路径 ====
os.makedirs("draw/pics", exist_ok=True)

# ==== 4. 绘图函数 ====
def plot_modified_bubble(df_all, title, save_path):
    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        df_all['Coverage'],
        df_all['Richness'],
        s=df_all['Scale'] * 3000,
        c=df_all['Authenticity'],
        cmap='Blues',
        alpha=0.8,
        edgecolors='black'
    )

    # 设置偏移
    # for i in range(len(df_all)):
    #     x = df_all['Coverage'].iloc[i]
    #     y = df_all['Richness'].iloc[i]
    #     dx = (i % 3 - 1) * 0.03   # -0.03, 0, +0.03
    #     dy = ((i // 3) % 3 - 1) * 0.03  # 随组别变化，避免重叠
    #     label = df_all['ShortLabel'].iloc[i]
    #     plt.text(
    #         x + dx, y + dy,
    #         label,
    #         fontsize=10,
    #         ha='center',
    #         va='center'
    #     )

    plt.xlabel('Coverage', fontsize=12)
    plt.ylabel('Richness', fontsize=12)
    # plt.title(title, fontsize=14)
    plt.grid(True)

    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Authenticity', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 图保存成功：{save_path}")

# ==== 5. 调用函数 ====
plot_modified_bubble(df, "SCAR Bubble (Cove vs. Richness, Color=Coverage)", "draw/pics/SCAR_bubble.png")
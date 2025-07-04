import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset, Dataset, Image
from torch.utils.data import DataLoader

# === 配置 ===
data_root = "data/COCO-caption"  # 替换为你本地 COCO 图像根路径
year = "2017"  # 2014 or 2017

# === Step 1: 加载标注（caption） ===
dataset = load_dataset(
    "jxie/coco_captions",
    # year,
    # split={"train": "train", "validation": "validation"}
)

# # === Step 2: 构造图像路径（从 image_id 构建） ===
# def construct_image_path(example, split):
#     folder = f"{'train2014' if split == 'train' else 'val2014'}"
#     file_name = f"COCO_{folder}_{int(example['image_id']):012d}.jpg"
#     full_path = os.path.join(data_root, folder, file_name)
#     return {"image": full_path}

# # === Step 3: 将图像路径绑定到 image 字段，并 decode 为 PIL.Image ===
# for split in ["train", "validation"]:
#     dataset[split] = dataset[split].map(
#         lambda x: construct_image_path(x, split),
#         remove_columns=["image"]  # 替换默认的 image 字段
#     )
#     dataset[split] = dataset[split].cast_column("image", Image(decode=True))

# # === Step 4: 验证图像和描述 ===
# sample = dataset["train"][0]
# print("Caption:", sample["caption"])
# sample["image"].show()

# # === Step 5: 可选：PyTorch DataLoader ===
# def collate_fn(batch):
#     images = [item["image"] for item in batch]
#     captions = [item["caption"] for item in batch]
#     return images, captions

# dataloader = DataLoader(dataset["train"], batch_size=16, shuffle=True, collate_fn=collate_fn)

# # === Step 6: 示例迭代 ===
# for batch_images, batch_captions in dataloader:
#     print("Batch size:", len(batch_images))
#     print("First caption:", batch_captions[0])
#     break
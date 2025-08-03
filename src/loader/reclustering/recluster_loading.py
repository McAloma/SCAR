import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
import random


def load_embeddings(data_dir):
    data = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(data_dir, fname), "r") as f:
                data.extend(json.load(f))
    return data


def regroup_by_id(data):
    return {item["id"]: np.array(item["embedding"]) for item in data}


def load_all_splits(src_root, name1, name2):
    """
    加载 train 和 val 的两个模态的数据
    返回结构:
    {
        'train': {mod1_data, mod2_data},
        'val':   {mod1_data, mod2_data},
        'all':   {mod1_dict, mod2_dict}
    }
    """
    result = {}
    splits = ["train", "val"]
    all_mod1, all_mod2 = {}, {}

    for split in splits:
        mod1_data = load_embeddings(os.path.join(src_root, split, name1))
        mod2_data = load_embeddings(os.path.join(src_root, split, name2))

        mod1_dict = regroup_by_id(mod1_data)
        mod2_dict = regroup_by_id(mod2_data)

        result[split] = {
            name1: mod1_dict,
            name2: mod2_dict
        }

        all_mod1.update(mod1_dict)
        all_mod2.update(mod2_dict)

    result["all"] = {
        name1: all_mod1,
        name2: all_mod2
    }

    return result


def recluster_and_swap_labels(train_mod1_embs, train_mod2_embs, val_mod1_embs, val_mod2_embs, k1, k2):
    print("[*] Clustering on train embeddings...")

    # Train embeddings
    ids1_train, vecs1_train = zip(*train_mod1_embs.items())
    ids2_train, vecs2_train = zip(*train_mod2_embs.items())

    # Fit on train
    kmeans1 = KMeans(n_clusters=k1).fit(np.vstack(vecs1_train))
    kmeans2 = KMeans(n_clusters=k2).fit(np.vstack(vecs2_train))

    # Predict train labels
    labels1_train = {i: int(label) for i, label in zip(ids1_train, kmeans1.labels_)}
    labels2_train = {i: int(label) for i, label in zip(ids2_train, kmeans2.labels_)}

    # Predict val labels
    ids1_val, vecs1_val = zip(*val_mod1_embs.items())
    ids2_val, vecs2_val = zip(*val_mod2_embs.items())

    val_labels1 = kmeans1.predict(np.vstack(vecs1_val))
    val_labels2 = kmeans2.predict(np.vstack(vecs2_val))

    labels1_val = {i: int(label) for i, label in zip(ids1_val, val_labels1)}
    labels2_val = {i: int(label) for i, label in zip(ids2_val, val_labels2)}

    return (labels1_train, labels2_train), (labels1_val, labels2_val)

def save_split(
    split_name,
    mod1_embs, mod2_embs,
    labels_from_2, labels_from_1,
    name1, name2,
    save_dir,
    batch_size=1024
):
    split_dir = os.path.join(save_dir, split_name)
    os.makedirs(os.path.join(split_dir, name1), exist_ok=True)
    os.makedirs(os.path.join(split_dir, name2), exist_ok=True)

    mod1_data = [{
        "embedding": emb.tolist(),
        "label": labels_from_2[uid],
        "type": name1,
        "id": uid
    } for uid, emb in mod1_embs.items()]

    mod2_data = [{
        "embedding": emb.tolist(),
        "label": labels_from_1[uid],
        "type": name2,
        "id": uid
    } for uid, emb in mod2_embs.items()]

    random.shuffle(mod1_data)
    random.shuffle(mod2_data)

    def save_batches(data, folder):
        for i in range(0, len(data), batch_size):
            with open(os.path.join(folder, f"batch_{i // batch_size}.json"), "w") as f:
                json.dump(data[i:i + batch_size], f)

    save_batches(mod1_data, os.path.join(split_dir, name1))
    save_batches(mod2_data, os.path.join(split_dir, name2))

    with open(os.path.join(split_dir, "label_counters.json"), "w") as f:
        json.dump({
            f"{name1}_labels_from_{name2}": dict(Counter([x["label"] for x in mod1_data])),
            f"{name2}_labels_from_{name1}": dict(Counter([x["label"] for x in mod2_data])),
            f"{name1}_size": len(mod1_data),
            f"{name2}_size": len(mod2_data),
        }, f, indent=2)


def main(args):
    all_data = load_all_splits(args.src_root, args.name1, args.name2)

    (labels1_train, labels2_train), (labels1_val, labels2_val) = recluster_and_swap_labels(
        all_data["train"][args.name1],
        all_data["train"][args.name2],
        all_data["val"][args.name1],
        all_data["val"][args.name2],
        k1=args.k1,
        k2=args.k2
    )

    # Save for train
    save_split(
        split_name="train",
        mod1_embs=all_data["train"][args.name1],
        mod2_embs=all_data["train"][args.name2],
        labels_from_2=labels2_train,
        labels_from_1=labels1_train,
        name1=args.name1,
        name2=args.name2,
        save_dir=args.tgt_root,
        batch_size=args.batch_size
    )

    # Save for val (using predicted labels)
    save_split(
        split_name="val",
        mod1_embs=all_data["val"][args.name1],
        mod2_embs=all_data["val"][args.name2],
        labels_from_2=labels2_val,
        labels_from_1=labels1_val,
        name1=args.name1,
        name2=args.name2,
        save_dir=args.tgt_root,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str, required=True, help="原始 embedding 根路径")
    parser.add_argument("--tgt_root", type=str, required=True, help="保存聚类结果的路径")
    parser.add_argument("--name1", type=str, required=True, help="模态1名称（如 audio/image/video）")
    parser.add_argument("--name2", type=str, required=True, help="模态2名称（如 text）")
    parser.add_argument("--k1", type=int, default=5)
    parser.add_argument("--k2", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    main(args)

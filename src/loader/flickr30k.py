from datasets import load_dataset

# 加载 Flickr8k 图文对
dataset = load_dataset("nlphuji/flickr30k", trust_remote_code=True)
print(dataset["train"][0])
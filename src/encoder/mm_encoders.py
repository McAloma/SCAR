import os, torch
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import CLIPProcessor, CLIPModel
from transformers import SiglipProcessor, SiglipModel
import open_clip
from PIL import Image
from typing import List, Union
from tqdm import tqdm



class CLIP_Encoder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)
        self.model.eval()

    def encode_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_embed = self.model.get_image_features(**inputs)
        return image_embed / image_embed.norm(dim=-1, keepdim=True)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_embeds = self.model.get_text_features(**inputs)
        return text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    def encode_images_batch(self, images: List[Union[str, Image.Image]], batch_size: int = 32) -> torch.Tensor:
        all_embeddings = []
        for i in tqdm(range(0, len(images), batch_size), desc="Encoding images"):
            batch = images[i:i+batch_size]
            batch_pil = [Image.open(p).convert("RGB") if isinstance(p, str) else p for p in batch]
            inputs = self.processor(images=batch_pil, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                embeds = self.model.get_image_features(**inputs)
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeds.cpu())
        return torch.cat(all_embeddings, dim=0)

    def encode_texts_batch(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch = texts[i:i+batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                embeds = self.model.get_text_features(**inputs)
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeds.cpu())
        return torch.cat(all_embeddings, dim=0)

    def compute_similarity(self, image_embed: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        return image_embed @ text_embeds.T
    








class SigLIP_Encoder:
    def __init__(self, model_name: str = "google/siglip-base-patch16-224", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SiglipModel.from_pretrained(model_name).to(self.device)
        self.processor = SiglipProcessor.from_pretrained(model_name, use_fast=True)
        self.model.eval()

    def encode_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_emb = self.model.get_image_features(**inputs)
        return image_emb / image_emb.norm(dim=-1, keepdim=True)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            text_emb = self.model.get_text_features(**inputs)
        return text_emb / text_emb.norm(dim=-1, keepdim=True)

    def encode_images_batch(self, images: List[Union[str, Image.Image]], batch_size: int = 32) -> torch.Tensor:
        all_embeds = []
        for i in tqdm(range(0, len(images), batch_size), desc="SigLIP: Encoding images"):
            batch = images[i:i+batch_size]
            batch_pil = [Image.open(p).convert("RGB") if isinstance(p, str) else p for p in batch]
            inputs = self.processor(images=batch_pil, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                embeds = self.model.get_image_features(**inputs)
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                all_embeds.append(embeds.cpu())
        return torch.cat(all_embeds, dim=0)

    def encode_texts_batch(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        all_embeds = []
        for i in tqdm(range(0, len(texts), batch_size), desc="SigLIP: Encoding texts"):
            batch = texts[i:i+batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                embeds = self.model.get_text_features(**inputs)
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                all_embeds.append(embeds.cpu())
        return torch.cat(all_embeds, dim=0)

    def compute_similarity(self, image_emb: torch.Tensor, text_embs: torch.Tensor) -> torch.Tensor:
        return image_emb @ text_embs.T







class CoCa_Encoder:
    def __init__(self, model_name: str = "coca_ViT-B-32", pretrained: str = "laion2B-s13B-b90k", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def encode_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_embed = self.model.encode_image(image_tensor)
        return image_embed / image_embed.norm(dim=-1, keepdim=True)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            text_embed = self.model.encode_text(tokens)
        return text_embed / text_embed.norm(dim=-1, keepdim=True)

    def encode_images_batch(self, images: List[Union[str, Image.Image]], batch_size: int = 32) -> torch.Tensor:
        all_embeddings = []
        for i in tqdm(range(0, len(images), batch_size), desc="CoCa: Encoding images"):
            batch = images[i:i + batch_size]
            batch_tensor = torch.stack([
                self.preprocess(Image.open(p).convert("RGB") if isinstance(p, str) else p)
                for p in batch
            ]).to(self.device)
            with torch.no_grad():
                embeds = self.model.encode_image(batch_tensor)
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeds.cpu())
        return torch.cat(all_embeddings, dim=0)

    def encode_texts_batch(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="CoCa: Encoding texts"):
            batch = texts[i:i + batch_size]
            tokens = self.tokenizer(batch).to(self.device)
            with torch.no_grad():
                embeds = self.model.encode_text(tokens)
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeds.cpu())
        return torch.cat(all_embeddings, dim=0)

    def compute_similarity(self, image_embed: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        return image_embed @ text_embeds.T











if __name__ == "__main__":
    image_paths = [f"data/CIFAR10/images/cifar10_1_32.jpg"] * 3
    text_list = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

    print("\nCLIP Encoder:")
    encoder = CLIP_Encoder()
    text_embs = encoder.encode_texts_batch(text_list)
    image_embs = encoder.encode_images_batch(image_paths)
    for i, img_emb in enumerate(image_embs):
        sim = encoder.compute_similarity(img_emb.unsqueeze(0), text_embs)
        print(f"Image {i} similarity:", sim)

    print("\nSigLIP Encoder:")
    encoder = SigLIP_Encoder()    
    text_embs = encoder.encode_texts_batch(text_list)
    image_embs = encoder.encode_images_batch(image_paths)
    for i, img_emb in enumerate(image_embs):
        sim = encoder.compute_similarity(img_emb.unsqueeze(0), text_embs)
        print(f"Image {i} similarity:", sim)

    print("\nCoCa Encoder:")
    encoder = CoCa_Encoder()    
    text_embs = encoder.encode_texts_batch(text_list)
    image_embs = encoder.encode_images_batch(image_paths)
    for i, img_emb in enumerate(image_embs):
        sim = encoder.compute_similarity(img_emb.unsqueeze(0), text_embs)
        print(f"Image {i} similarity:", sim)
import os, torch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from transformers import ViTFeatureExtractor, ViTModel
from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoModel



class ResNet_Encoder:
    def __init__(self):
        model_name = "microsoft/resnet-50"
        cache_dir = "./ckpts/resnet"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(self.device)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.eval()
       
    def encode(self, image):
        inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state 
            pooled = torch.nn.functional.adaptive_avg_pool2d(hidden, 1)
            image_embedding = pooled.view(pooled.size(0), -1).cpu().numpy() 
        return image_embedding
    
    def encode_batch(self, images):
        inputs = self.feature_extractor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state 
            pooled = torch.nn.functional.adaptive_avg_pool2d(hidden, 1)
            image_embedding = pooled.view(pooled.size(0), -1).cpu().numpy()
        return image_embedding


class ViT_Encoder:
    def __init__(self):
        model_name="google/vit-base-patch16-224-in21k"
        cache_dir = "./ckpts/vit"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViTModel.from_pretrained(model_name, cache_dir=cache_dir).to(self.device)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.eval()

    def encode(self, image):
        inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        image_embedding = outputs.last_hidden_state[:, 0].cpu().numpy()
        return image_embedding

    def encode_batch(self, images):
        inputs = self.feature_extractor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        image_embedding = outputs.last_hidden_state[:, 0].cpu().numpy()
        return image_embedding


class DINO_Encoder:
    def __init__(self):
        model_name = "facebook/dinov2-base"
        cache_dir = "./ckpts/dino"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.eval()

    def encode(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        image_embedding = outputs.last_hidden_state[:, 0].cpu().numpy()
        return image_embedding
    
    def encode_batch(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        image_embedding = outputs.last_hidden_state[:, 0].cpu().numpy()
        return image_embedding



if __name__ == "__main__":
    from PIL import Image
    img_path = f"data/CIFAR10/images/cifar10_0_35.jpg"
    img = Image.open(img_path).convert("RGB")
    imgs = [img] * 500

    resnet_encoder = ResNet_Encoder()
    image_embedding = resnet_encoder.encode(img)
    print("Size of Resnet Encoding: ", image_embedding.shape)  # Should be (1, 2048) for ResNet-50
    image_embedding = resnet_encoder.encode_batch(imgs)
    print("Size of Resnet Encoding images: ", image_embedding.shape)  # Should be (500, 2048) for ResNet-50

    vit_encoder = ViT_Encoder()
    image_embedding = vit_encoder.encode(img)
    print("Size of ViT Encoding: ", image_embedding.shape)  # Should be (1, 768) for ViT-base
    image_embedding = vit_encoder.encode_batch(imgs)
    print("Size of ViT Encoding images: ", image_embedding.shape)  # Should be (500, 768) for ViT-base

    dino_encoder = DINO_Encoder()
    image_embedding = dino_encoder.encode(img)
    print("Size of DINO Encoding: ", image_embedding.shape)  # Should be (1, 768) for DINO
    image_embedding = dino_encoder.encode_batch(imgs)
    print("Size of DINO Encoding images: ", image_embedding.shape)  # Should be (500, 768) for DINO


    # ———————————— time test ————————————

    # from time import time
    # resnet_encoder = ResNet_Encoder()

    # begin = time()
    # image_embedding = resnet_encoder.encode(img)
    # print("Size of Resnet Encoding: ", image_embedding.shape)  # Should be (1, 2048) for ResNet-50
    # end = time()
    # print("Time taken for single image: ", end - begin)

    # begin = time()
    # image_embedding = resnet_encoder.encode_batch(imgs)
    # print("Size of Resnet Encoding: ", image_embedding.shape)  # Should be (10, 2048) for ResNet-50
    # end = time()
    # print("Time taken for batch of images: ", end - begin)

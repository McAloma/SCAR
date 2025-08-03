import sys, os, av, cv2
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/SCAR_data_description/")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import numpy as np
from transformers import AutoProcessor, AutoModel
from typing import List, Union
from tqdm import tqdm



class XCLIP_Encoder:
    def __init__(self, model_name: str = "microsoft/xclip-base-patch32", device: str = None, clip_len: int = 8, frame_sample_rate: int = 1):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate

    def _read_video_pyav(self, file_path: str, indices: List[int]) -> np.ndarray:
        container = av.open(file_path)
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    def _sample_frame_indices(self, seg_len: int) -> List[int]:
        converted_len = int(self.clip_len * self.frame_sample_rate)
        if seg_len < converted_len:
            # if video shorter than needed frames, sample as many as possible
            indices = np.linspace(0, seg_len - 1, num=seg_len).astype(np.int64)
        else:
            end_idx = np.random.randint(converted_len, seg_len)
            start_idx = end_idx - converted_len
            indices = np.linspace(start_idx, end_idx, num=self.clip_len)
            indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices.tolist()

    def encode_video(self, video_path: str) -> torch.Tensor:
        container = av.open(video_path)
        seg_len = container.streams.video[0].frames
        indices = self._sample_frame_indices(seg_len)
        frames = self._read_video_pyav(video_path, indices)
        inputs = self.processor(videos=list(frames), return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            video_embeds = self.model.get_video_features(**inputs)
        video_embeds = video_embeds / video_embeds.norm(dim=-1, keepdim=True)
        # print(f"[Video Embed] shape: {video_embeds.shape}")
        return video_embeds

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            text_embeds = self.model.get_text_features(**inputs)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        # print(f"[Text Embed] shape: {text_embeds.shape}")
        return text_embeds

    def encode_videos_batch(self, video_paths: List[str], batch_size: int = 4) -> torch.Tensor:
        all_embeds = []
        for i in tqdm(range(0, len(video_paths), batch_size), desc="Encoding videos batch"):
            batch_paths = video_paths[i:i+batch_size]
            batch_frames = []
            for vp in batch_paths:
                container = av.open(vp)
                seg_len = container.streams.video[0].frames
                indices = self._sample_frame_indices(seg_len)
                frames = self._read_video_pyav(vp, indices)
                batch_frames.append(list(frames))
            inputs = self.processor(videos=batch_frames, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                embeds = self.model.get_video_features(**inputs)
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                all_embeds.append(embeds.cpu())
        all_embeds = torch.cat(all_embeds, dim=0)
        # print(f"[Batch Video Embeds] shape: {all_embeds.shape}")
        return all_embeds

    def encode_texts_batch(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        all_embeds = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts batch"):
            batch = texts[i:i+batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                embeds = self.model.get_text_features(**inputs)
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                all_embeds.append(embeds.cpu())
        all_embeds = torch.cat(all_embeds, dim=0)
        # print(f"[Batch Text Embeds] shape: {all_embeds.shape}")
        return all_embeds

    def compute_similarity(self, video_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        return video_embeds @ text_embeds.T





from src.library.VideoCLIP.modeling import VideoCLIP_XL
from src.library.VideoCLIP.utils.text_encoder import text_encoder




class VideoCLIP_Encoder:
    def __init__(self, model_path: str = "src/library/VideoCLIP/VideoCLIP-XL.bin", device: str = None, clip_len: int = 8):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_len = clip_len
        self.model = VideoCLIP_XL().to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        self.v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        return (data / 255.0 - self.v_mean) / self.v_std

    def _read_and_process_video(self, video_path: str) -> torch.Tensor:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames found in video: {video_path}")

        step = max(len(frames) // self.clip_len, 1)
        sampled = frames[::step][:self.clip_len]

        processed = []
        for fr in sampled:
            fr = fr[:, :, ::-1]  # BGR to RGB
            fr = cv2.resize(fr, (224, 224))
            fr = np.expand_dims(self._normalize(fr), axis=(0, 1))  # (1,1,H,W,C)
            processed.append(fr)

        vid_tube = np.concatenate(processed, axis=1)  # (1, clip_len, H, W, C)
        vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))  # (B, T, C, H, W)
        vid_tensor = torch.from_numpy(vid_tube).float().to(self.device)
        return vid_tensor

    def encode_video(self, video_path: str) -> torch.Tensor:
        with torch.no_grad():
            video_tensor = self._read_and_process_video(video_path)
            feats = self.model.vision_model.get_vid_features(video_tensor).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats  # shape: (1, D)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            tokenized = text_encoder.tokenize(texts, truncate=True).to(self.device)
            feats = self.model.text_model.encode_text(tokenized).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats  # shape: (len(texts), D)

    def encode_videos_batch(self, video_paths: List[str], batch_size: int = 1) -> torch.Tensor:
        all_feats = []
        for i in tqdm(range(0, len(video_paths), batch_size), desc="Encoding video batch"):
            batch = video_paths[i:i + batch_size]
            batch_feats = [self.encode_video(p) for p in batch]
            all_feats.append(torch.cat(batch_feats, dim=0).cpu())
        return torch.cat(all_feats, dim=0)

    def encode_texts_batch(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        all_feats = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding text batch"):
            batch = texts[i:i + batch_size]
            feats = self.encode_text(batch)
            all_feats.append(feats.cpu())
        return torch.cat(all_feats, dim=0)

    def compute_similarity(self, video_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        video_embeds = video_embeds.to(self.device)
        text_embeds = text_embeds.to(self.device)
        return text_embeds @ video_embeds.T






if __name__ == "__main__":
    # encoder = XCLIP_Encoder()

    # video_embed = encoder.encode_video("data/MSR_VTT/MSRVTT_Videos/video/video1.mp4")       # (n, 512)
    # text_embeds = encoder.encode_text(["A man is singing.", "A cat is walking."])           # (n, 512)

    # sim = encoder.compute_similarity(video_embed, text_embeds)
    # print(video_embed.shape, text_embeds.shape)
    # print(sim)



    encoder = VideoCLIP_Encoder()
    video_embed = encoder.encode_video("data/MSR_VTT/MSRVTT_Videos/video/video1.mp4")     # shape: (1, 768)
    text_embeds = encoder.encode_texts_batch(["A man is singing.", "A cat is walking."])  # shape: (2, 768)

    sim = encoder.compute_similarity(video_embed, text_embeds)
    print(video_embed.shape, text_embeds.shape)
    print(sim)
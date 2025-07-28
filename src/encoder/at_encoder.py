import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from typing import List, Union
import torch
import torchaudio
import numpy as np
from transformers import ClapProcessor, ClapModel, ClapConfig
from tqdm import tqdm
from datasets import Dataset, load_dataset




class CLAP_Encoder:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        local_model_path = "ckpts/clap"
        self.processor = ClapProcessor.from_pretrained(local_model_path)
        self.model = ClapModel.from_pretrained(local_model_path).to(self.device)
        self.model.eval()

    def encode_audio(self, audio_path: str) -> torch.Tensor:
        waveform, sr = torchaudio.load(audio_path)
        return self.encode_audio_from_data(waveform, sr)

    def encode_audio_from_data(self, waveform: Union[torch.Tensor, 'np.ndarray'], sampling_rate: int) -> torch.Tensor:
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform, dtype=torch.float32)
        else:
            waveform = waveform.to(torch.float32)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        print(f"waveform.shape={waveform.shape}, waveform.dtype={waveform.dtype}, sampling_rate={sampling_rate}")

        if sampling_rate != 48000:
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=48000)
            waveform = resampler(waveform)
            sampling_rate = 48000

        waveform_np = waveform.squeeze(0).numpy() 

        inputs = self.processor(audios=waveform_np, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            audio_outputs = self.model.get_audio_features(**inputs)
        audio_embed = audio_outputs / audio_outputs.norm(dim=-1, keepdim=True)
        return audio_embed.squeeze(0)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_outputs = self.model.get_text_features(**inputs)
        text_embeds = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
        return text_embeds

    def encode_audios_from_dataset(self, audio_arrays: List[np.ndarray], sampling_rates: List[int], batch_size: int = 8) -> torch.Tensor:
        all_embeds = []
        for i in tqdm(range(0, len(audio_arrays), batch_size), desc="Encoding audio batches"):
            batch_waveforms = []

            for waveform, sr in zip(audio_arrays[i:i+batch_size], sampling_rates[i:i+batch_size]):
                # Convert to float32 tensor and ensure shape [1, num_samples]
                waveform = torch.tensor(waveform, dtype=torch.float32)
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                if sr != 48000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=48000)
                    waveform = resampler(waveform)
                    sr = 48000
                waveform_np = waveform.squeeze(0).numpy()
                batch_waveforms.append(waveform_np)

            # 👇 注意传 numpy list，不能传 torch.Tensor
            inputs = self.processor(audios=batch_waveforms, sampling_rate=48000, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                embeds = self.model.get_audio_features(**inputs)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            all_embeds.append(embeds.cpu())

        return torch.cat(all_embeds, dim=0)

    def encode_texts_batch(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        all_embeds = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts batch"):
            batch = texts[i:i+batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                embeds = self.model.get_text_features(**inputs)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            all_embeds.append(embeds.cpu())
        return torch.cat(all_embeds, dim=0)

    def compute_similarity(self, audio_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        return audio_embeds @ text_embeds.T


    

if __name__ == "__main__":
    dataset = load_dataset("OpenSound/AudioCaps", cache_dir="data/Audiocaps")

    sample = dataset["train"][0]

    waveform = sample["audio"]["array"]
    sampling_rate = sample["audio"]["sampling_rate"]
    caption = sample["caption"]

    encoder = CLAP_Encoder()

    audio_embedding = encoder.encode_audio_from_data(waveform, sampling_rate)
    print("Audio embedding shape:", audio_embedding.shape)

    text_embedding = encoder.encode_text([caption])
    print("Text embedding shape:", text_embedding.shape)

    similarity = encoder.compute_similarity(audio_embedding.unsqueeze(0), text_embedding)
    print("Similarity:", similarity.item())


    print("Batch encoding example:")
    
    subset = dataset["train"].select(range(100))
    audio_arrays = [sample["audio"]["array"] for sample in subset]
    sampling_rates = [sample["audio"]["sampling_rate"] for sample in subset]
    audio_embeddings = encoder.encode_audios_from_dataset(audio_arrays, sampling_rates, batch_size=64)

    print("Audio embedding shape:", audio_embeddings.shape)  # shape: [100, D]

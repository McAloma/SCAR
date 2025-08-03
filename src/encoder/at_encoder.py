import sys, os
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/SCAR_data_description/")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from typing import List, Union
import torch
import torchaudio
import tensorflow as tf
import numpy as np
import torch.nn.functional as F
from transformers import ClapProcessor, ClapModel, ClapConfig
from tqdm import tqdm
from datasets import Dataset, load_dataset

from collections import OrderedDict




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






import yaml, random, argparse
import torchaudio.transforms as T
from src.library.Pengi.models.pengi import PENGI
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader


class AudioDataset(Dataset):
    def __init__(self, waveforms, sampling_rates, target_sr, expected_len):
        self.waveforms = waveforms
        self.sampling_rates = sampling_rates
        self.target_sr = target_sr
        self.expected_len = expected_len

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        waveform = self.waveforms[idx]
        sr = self.sampling_rates[idx]

        if isinstance(waveform, np.ndarray):
            waveform = torch.tensor(waveform, dtype=torch.float32)
        else:
            waveform = waveform.to(torch.float32)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # [1, T]

        # Resample
        if sr != self.target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        # Pad / Crop
        if waveform.shape[1] < self.expected_len:
            repeat_factor = int(np.ceil(self.expected_len / waveform.shape[1]))
            waveform = waveform.repeat(1, repeat_factor)[:, :self.expected_len]
        else:
            start = random.randint(0, waveform.shape[1] - self.expected_len)
            waveform = waveform[:, start : start + self.expected_len]

        return waveform


class PengiAudioTextEncoder:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.config_path = "src/library/Pengi/configs/base.yml"
        self.model_path = "ckpts/pengi/base.pth"

        self.model, self.enc_tokenizer, self.dec_tokenizer, self.args = self.get_model_and_tokenizer(config_path=self.config_path)
        self.model.eval()

    def read_config_as_args(self,config_path):
        return_dict = {}
        with open(config_path, "r") as f:
            yml_config = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in yml_config.items():
            return_dict[k] = v
        return argparse.Namespace(**return_dict)

    def get_model_and_tokenizer(self, config_path):
        args = self.read_config_as_args(config_path)
        args.prefix_dim = args.d_proj
        args.total_prefix_length = 2*args.prefix_length
        if not args.use_text_model:
            args.text_model = args.text_decoder

        # Copy relevant configs from dataset_config
        args.sampling_rate = args.dataset_config['sampling_rate']
        args.duration = args.dataset_config['duration']

        model = PENGI(
            audioenc_name=args.audioenc_name,
            sample_rate=args.sampling_rate,
            window_size=args.window_size,
            hop_size=args.hop_size,
            mel_bins=args.mel_bins,
            fmin=args.fmin,
            fmax=args.fmax,
            classes_num=None,
            out_emb=args.out_emb,
            specaug=args.specaug,
            mixup=args.mixup,
            use_text_encoder=args.use_text_model,
            text_encoder=args.text_model,
            text_encoder_embed_dim=args.transformer_embed_dim,
            freeze_text_encoder_weights=args.freeze_text_encoder_weights,
            text_decoder=args.text_decoder,
            prefix_length=args.prefix_length,
            clip_length=args.prefix_length_clip,
            prefix_size=args.prefix_dim,
            num_layers=args.num_layers,
            normalize_prefix=args.normalize_prefix,
            mapping_type=args.mapping_type,
            freeze_text_decoder_weights=args.freeze_gpt_weights,
            d_proj=args.d_proj,
            use_pretrained_audioencoder=args.use_pretrained_audioencoder,
            freeze_audio_encoder_weights=args.freeze_audio_encoder_weights,
            use_precomputed_melspec=False,
            pretrained_audioencoder_path=None,
        )
        model.enc_text_len = args.dataset_config['enc_text_len']
        model.dec_text_len = args.dataset_config['dec_text_len']
        model_state_dict = torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True)['model']

        model.load_state_dict(model_state_dict, strict=False)
        
        enc_tokenizer = AutoTokenizer.from_pretrained(args.text_model)
        if 'gpt' in args.text_model:
            enc_tokenizer.add_special_tokens({'pad_token': '!'})

        dec_tokenizer = AutoTokenizer.from_pretrained(args.text_decoder)
        if 'gpt' in args.text_decoder:
            dec_tokenizer.add_special_tokens({'pad_token': '!'})

        model = model.to(self.device)
        
        return model, enc_tokenizer, dec_tokenizer, args

    def _preprocess_audio(self, audio_path: str, resample: bool = True) -> torch.Tensor:
        waveform, sr = torchaudio.load(audio_path)
        target_sr = self.args.sampling_rate
        if resample and sr != target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
        waveform = waveform.reshape(-1)

        expected_len = int(target_sr * self.args.duration)
        if waveform.shape[0] < expected_len:
            repeat_factor = int(np.ceil(expected_len / waveform.shape[0]))
            waveform = waveform.repeat(repeat_factor)[:expected_len]
        else:
            start = random.randint(0, waveform.shape[0] - expected_len)
            waveform = waveform[start : start + expected_len]

        waveform = waveform.unsqueeze(0).to(self.device)
        return waveform

    def encode_audio_from_data(self, waveform: Union[np.ndarray, torch.Tensor], sampling_rate: int) -> torch.Tensor:
        if isinstance(waveform, np.ndarray):
            waveform = torch.tensor(waveform, dtype=torch.float32)
        else:
            waveform = waveform.to(torch.float32)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        target_sr = self.args.sampling_rate
        if sampling_rate != target_sr:
            resampler = T.Resample(orig_freq=sampling_rate, new_freq=target_sr)
            waveform = resampler(waveform)

        expected_len = int(target_sr * self.args.duration)
        if waveform.shape[1] < expected_len:
            repeat_factor = int(np.ceil(expected_len / waveform.shape[1]))
            waveform = waveform.repeat(1, repeat_factor)[:, :expected_len]
        else:
            start = random.randint(0, waveform.shape[1] - expected_len)
            waveform = waveform[:, start : start + expected_len]

        waveform = waveform.to(self.device)

        self.model.eval()
        with torch.no_grad():
            embedding = self.model.audio_encoder(waveform)[0]
            if self.args.normalize_prefix:
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu()

    def encode_text(self, texts: List[str], use_encoder: bool = True) -> torch.Tensor:
        tokenizer = self.enc_tokenizer if use_encoder else self.dec_tokenizer
        max_length = self.model.enc_text_len if use_encoder else self.model.dec_text_len

        inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self.model.eval()
        with torch.no_grad():
            if use_encoder:
                embeddings = self.model.caption_encoder(inputs)
            else:
                input_ids = inputs["input_ids"]
                embeddings = self.model.caption_decoder.gpt.transformer.wte(input_ids)
        return embeddings.cpu()
    
    def encode_audios_from_dataset(
        self, 
        waveforms: List[Union[np.ndarray, torch.Tensor]], 
        sampling_rates: List[int],
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> torch.Tensor:
        """
        Parallel and batch version of encode_audio_from_data
        """
        target_sr = self.args.sampling_rate
        expected_len = int(target_sr * self.args.duration)

        print("[Audio] Preprocessing + Batching waveforms...")
        dataset = AudioDataset(waveforms, sampling_rates, target_sr, expected_len)
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=True
        )

        all_embeddings = []
        self.model.eval()
        print("[Audio] Encoding audio batches...")
        with torch.no_grad():
            for batch in tqdm(loader, desc="Encoding Audio Batches"):
                batch = batch.to(self.device)           # [B, 1, T]
                batch = batch.squeeze(1)                # → [B, T]
                embeds = self.model.audio_encoder(batch)[0]
                if self.args.normalize_prefix:
                    embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeds.cpu())

        return torch.cat(all_embeddings, dim=0)  # [N, D]

    def encode_texts_batch(
        self, 
        texts: List[str], 
        use_encoder: bool = True, 
        batch_size: int = 32
    ) -> torch.Tensor:
        """Batch version of encode_text with manual batching"""
        
        tokenizer = self.enc_tokenizer if use_encoder else self.dec_tokenizer
        max_length = self.model.enc_text_len if use_encoder else self.model.dec_text_len

        all_embeddings = []
        self.model.eval()
        print("[Text] Encoding text batches...")
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Text Batches"):
                batch_texts = texts[i:i + batch_size]
                inputs = tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=max_length, 
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                if use_encoder:
                    embeddings = self.model.caption_encoder(inputs)  # [B, D]
                else:
                    input_ids = inputs["input_ids"]
                    embeddings = self.model.caption_decoder.gpt.transformer.wte(input_ids)  # [B, T, D]

                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def compute_similarity(self, audio_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        audio_embeds = F.normalize(audio_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        return audio_embeds @ text_embeds.T







if __name__ == "__main__":
    dataset = load_dataset("OpenSound/AudioCaps", cache_dir="data/Audiocaps")

    sample = dataset["train"][0]

    waveform = sample["audio"]["array"]
    sampling_rate = sample["audio"]["sampling_rate"]
    caption = sample["caption"]


    # # CALP
    # encoder = CLAP_Encoder()

    # audio_embedding = encoder.encode_audio_from_data(waveform, sampling_rate)
    # print("Audio embedding shape:", audio_embedding.shape)

    # text_embedding = encoder.encode_text([caption])
    # print("Text embedding shape:", text_embedding.shape)

    # similarity = encoder.compute_similarity(audio_embedding.unsqueeze(0), text_embedding)
    # print("Similarity:", similarity.item())


    # print("Batch encoding example:")
    
    # subset = dataset["train"].select(range(100))
    # audio_arrays = [sample["audio"]["array"] for sample in subset]
    # sampling_rates = [sample["audio"]["sampling_rate"] for sample in subset]
    # audio_embeddings = encoder.encode_audios_from_dataset(audio_arrays, sampling_rates, batch_size=64)

    # print("Audio embedding shape:", audio_embeddings.shape)  # shape: [100, D]



    # Pengi
    encoder = PengiAudioTextEncoder(device="cuda")

    audio_embedding = encoder.encode_audio_from_data(waveform, sampling_rate)   # shape: [1, 1024]
    text_embeddings = encoder.encode_text([caption], use_encoder=True)          # shape: [1, 1024]
    similarity = encoder.compute_similarity(audio_embedding, text_embeddings)
    print(similarity)

    subset = dataset["train"].select(range(100))
    audio_arrays = [sample["audio"]["array"] for sample in subset]
    sampling_rates = [sample["audio"]["sampling_rate"] for sample in subset]
    audio_embeddings = encoder.encode_audios_from_dataset(audio_arrays, sampling_rates, batch_size=64)

    print("Audio embedding shape:", audio_embeddings.shape)  # shape: [100, D]

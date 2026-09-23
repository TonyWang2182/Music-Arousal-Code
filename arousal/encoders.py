"""Original-gain audio decoding, 32-D MIR and genuine frozen MERT embeddings."""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from scipy import signal

MERT_MODEL = "m-a-p/MERT-v1-95M"
MERT_SHA = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"


def load_audio(path: Path, sample_rate: int = 24000) -> np.ndarray:
    """Decode the whole file, downmix channels, resample, never normalize gain.

    The MERT checkpoint's own feature extractor may normalize each input chunk;
    that pretrained preprocessing is separate and is not altered here.
    """
    import soundfile as sf
    wave, native_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if not len(wave) or not np.isfinite(wave).all() or native_rate <= 0:
        raise ValueError(f"Empty, invalid or nonfinite audio: {path}")
    wave = wave.mean(axis=1, dtype=np.float64).astype(np.float32)
    if native_rate != sample_rate:
        divisor = math.gcd(int(native_rate), int(sample_rate))
        wave = signal.resample_poly(wave, sample_rate // divisor, native_rate // divisor)
    if not np.isfinite(wave).all():
        raise ValueError(f"Nonfinite resampled audio: {path}")
    return np.asarray(wave, dtype=np.float32)


def handcrafted(y: np.ndarray, sr: int = 24000) -> np.ndarray:
    """32 dimensions: mean/std of ten scalar tracks plus mean 12-bin chroma.

    RMS, ZCR, centroid, bandwidth, 85% rolloff, flatness, positive spectral
    flux, low/mid/high power fractions. STFT: 1024-sample Hann, hop 256.
    This compact baseline is not the official MARBLE implementation.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or sr <= 0:
        raise ValueError("Invalid audio for MIR extraction")
    if len(y) < 1024:
        y = np.pad(y, (0, 1024 - len(y)))
    frames = np.lib.stride_tricks.sliding_window_view(y, 1024)[::256]
    mag = np.abs(np.fft.rfft(frames * np.hanning(1024), axis=1)) + 1e-12
    freqs = np.fft.rfftfreq(1024, 1 / sr)
    power = mag ** 2
    mass = power.sum(axis=1)
    centroid = (mag * freqs).sum(axis=1) / mag.sum(axis=1)
    bandwidth = np.sqrt((mag * (freqs - centroid[:, None]) ** 2).sum(axis=1) / mag.sum(axis=1))
    rolloff = freqs[np.argmax(np.cumsum(power, axis=1) >= .85 * mass[:, None], axis=1)]
    norm = mag / mag.sum(axis=1, keepdims=True)
    flux = np.r_[0, np.maximum(np.diff(norm, axis=0), 0).sum(axis=1)]
    curves = [np.sqrt((frames ** 2).mean(axis=1)),
              (np.diff(np.signbit(frames), axis=1) != 0).mean(axis=1),
              centroid / sr, bandwidth / sr, rolloff / sr,
              np.exp(np.log(power).mean(axis=1)) / power.mean(axis=1), flux]
    curves += [power[:, mask].sum(axis=1) / mass for mask in
               (freqs < 250, (freqs >= 250) & (freqs < 2000), freqs >= 2000)]
    stats = np.array([[curve.mean(), curve.std()] for curve in curves]).ravel()
    chroma = np.zeros((len(frames), 12))
    for index, freq in enumerate(freqs):
        if 40 <= freq <= 5000:
            pitch_class = int(round(69 + 12 * np.log2(freq / 440))) % 12
            chroma[:, pitch_class] += power[:, index]
    chroma /= np.maximum(chroma.sum(axis=1, keepdims=True), 1e-12)
    features = np.r_[stats, chroma.mean(axis=0)].astype(np.float64)
    if features.shape != (32,) or not np.isfinite(features).all():
        raise ValueError("Invalid MIR feature vector")
    return features


def chunk_ranges(n_samples: int, chunk_size: int, min_size: int) -> list[tuple[int, int]]:
    """Nonoverlapping chunks; an unencodable short tail joins its predecessor."""
    if n_samples < min_size or min_size <= 0 or chunk_size < min_size:
        raise ValueError("Audio/chunk size is below the MERT convolution receptive field")
    ranges = [(start, min(start + chunk_size, n_samples)) for start in range(0, n_samples, chunk_size)]
    if len(ranges) > 1 and ranges[-1][1] - ranges[-1][0] < min_size:
        ranges[-2] = (ranges[-2][0], ranges[-1][1])
        ranges.pop()
    return ranges


class MERTEncoder:
    """Strict checkpoint loading and output-frame-weighted full-clip pooling."""

    def __init__(self, model_id: str = MERT_MODEL, revision: str = MERT_SHA,
                 device: str = "cpu", chunk_seconds: float = 5.0):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("MERT revision must be an exact 40-character commit SHA")
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, AutoModel, Wav2Vec2FeatureExtractor

        self.torch, self.device = torch, device
        self.model_id, self.revision = model_id, revision
        config = AutoConfig.from_pretrained(model_id, revision=revision, code_revision=revision, trust_remote_code=True)
        model = AutoModel.from_config(config, trust_remote_code=True, code_revision=revision)
        weights_path = hf_hub_download(model_id, "pytorch_model.bin", revision=revision)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        expected = model.state_dict()
        mapped = {}
        for key, value in state.items():
            newkey = key
            for old, new in ((".weight_g", ".parametrizations.weight.original0"),
                             (".weight_v", ".parametrizations.weight.original1")):
                if key.endswith(old) and key[:-len(old)] + new in expected:
                    newkey = key[:-len(old)] + new
            if newkey in mapped:
                raise ValueError(f"Ambiguous weight conversion: {key}")
            mapped[newkey] = value
        model.load_state_dict(mapped, strict=True)
        self.model = model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, revision=revision)
        self.sr = int(self.processor.sampling_rate)
        if self.sr != 24000:
            raise ValueError("Unexpected MERT sample rate; this protocol requires 24000 Hz")
        self.chunk_seconds = float(chunk_seconds)
        # Minimum input needed to yield a convolution frame, from the checkpoint.
        receptive_field = 1
        for kernel, stride in reversed(list(zip(config.conv_kernel, config.conv_stride))):
            receptive_field = (receptive_field - 1) * stride + kernel
        self.min_samples = receptive_field
        self.metadata = {"model_id": model_id, "revision": revision,
                         "processor": self.processor.to_dict(), "device": device,
                         "chunk_seconds": self.chunk_seconds, "min_samples": self.min_samples,
                         "gain_policy": "original waveform; checkpoint processor preprocessing retained",
                         "pooling": "sum over all output frames / total actual output frames",
                         "tail_policy": "merge tail smaller than convolution receptive field into previous chunk",
                         "weight_loading": "strict, historical weight_norm key mapping"}

    def encode(self, wave: np.ndarray, sr: int = 24000) -> np.ndarray:
        wave = np.asarray(wave, dtype=np.float32)
        if wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
            raise ValueError("Invalid audio for MERT")
        if sr != self.sr:
            divisor = math.gcd(sr, self.sr)
            wave = signal.resample_poly(wave, self.sr // divisor, sr // divisor).astype(np.float32)
        sums, count = None, 0
        for start, stop in chunk_ranges(len(wave), int(self.chunk_seconds * self.sr), self.min_samples):
            inputs = self.processor(wave[start:stop], sampling_rate=self.sr, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                states = self.model(**inputs, output_hidden_states=True).hidden_states
            lengths = {int(state.shape[1]) for state in states}
            if len(lengths) != 1 or next(iter(lengths)) <= 0:
                raise ValueError("MERT layers have inconsistent or empty output frame counts")
            pooled = np.stack([state[0].double().sum(0).cpu().numpy() for state in states])
            sums = pooled if sums is None else sums + pooled
            count += next(iter(lengths))
        features = (sums / count).astype(np.float32)
        if features.ndim != 2 or features.shape[0] < 13 or features.shape[1] != 768 or not np.isfinite(features).all():
            raise ValueError(f"Invalid MERT embedding shape or values: {features.shape}")
        return features

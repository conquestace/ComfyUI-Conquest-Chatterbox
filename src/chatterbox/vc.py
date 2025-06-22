# vc.py  — Chatterbox Voice-Conversion for ComfyUI
from pathlib import Path
import librosa
import torch
import perth
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.s3tokenizer import S3_SR
from .models.s3gen      import S3GEN_SR, S3Gen

REPO_ID = "ResembleAI/chatterbox"


class ChatterboxVC:
    """One-shot voice conversion using S3Gen."""
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(self, s3gen: S3Gen, device: str, ref_dict: dict | None = None):
        self.sr   = S3GEN_SR
        self.device = device
        self.s3gen = s3gen.eval().to(device)
        self.watermarker = perth.PerthImplicitWatermarker()

        # Move tensors in ref-dict onto the chosen device
        if ref_dict is None:
            self.ref_dict = None
        else:
            self.ref_dict = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in ref_dict.items()
            }

    # ------------------------------------------------------------------ #
    #  Loading helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def _load_checkpoint_dir(cls, ckpt_dir: Path, device: str):
        # Handle CUDA vs CPU/MPS safely (models saved with CUDA tensors)
        map_loc = torch.device("cpu") if device in {"cpu", "mps"} else None

        # Optional built-in voice
        ref_dict = None
        builtin = ckpt_dir / "conds.pt"
        if builtin.exists():
            states   = torch.load(builtin, map_location=map_loc)
            ref_dict = states["gen"]

        # S3Gen weights (safetensors)
        s3gen = S3Gen()
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)

        return cls(s3gen, device, ref_dict)

    @classmethod
    def from_local(cls, ckpt_dir: str | Path, device: str = "cuda"):
        return cls._load_checkpoint_dir(Path(ckpt_dir), device)

    @classmethod
    def from_pretrained(cls, device: str = "cuda"):
        # macOS M-series fallback
        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"

        # Download required files once to HF-cache
        for fname in ("s3gen.safetensors", "conds.pt"):
            hf_hub_download(repo_id=REPO_ID, filename=fname)

        ckpt_root = Path(hf_hub_download(repo_id=REPO_ID, filename="s3gen.safetensors")).parent
        return cls._load_checkpoint_dir(ckpt_root, device)

    # ------------------------------------------------------------------ #
    #  Reference-voice utilities
    # ------------------------------------------------------------------ #
    def set_target_voice(self, wav_path: str | Path):
        """Extract a new reference embedding from `wav_path`."""
        ref_wav, _ = librosa.load(wav_path, sr=S3GEN_SR)
        ref_wav    = ref_wav[: self.DEC_COND_LEN]
        self.ref_dict = self.s3gen.embed_ref(ref_wav, S3GEN_SR, device=self.device)

    # ------------------------------------------------------------------ #
    #  Main conversion API
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def generate(self, audio_path: str | Path, target_voice_path: str | Path | None = None):
        """
        Convert `audio_path` into the timbre of the target voice.
        If `target_voice_path` is None, uses the last set (or built-in) voice.
        """
        if target_voice_path:
            self.set_target_voice(target_voice_path)
        assert self.ref_dict is not None, "No reference voice available!"

        # 1. Load source audio at 16 kHz (tokenizer SR)
        src_wav, _ = librosa.load(audio_path, sr=S3_SR)
        src_tensor = torch.from_numpy(src_wav).float().to(self.device).unsqueeze(0)  # (1,T)

        # 2. Tokenize → S3 tokens
        s3_tokens, _ = self.s3gen.tokenizer(src_tensor)

        # 3. Run S3Gen decoder with reference conditioning
        out_wav, _ = self.s3gen.inference(speech_tokens=s3_tokens, ref_dict=self.ref_dict)
        out_wav = out_wav.squeeze(0).cpu().numpy()

        # 4. Watermark & return (B,C,T) tensor for Comfy
        out_wav = self.watermarker.apply_watermark(out_wav, sample_rate=self.sr)
        return torch.from_numpy(out_wav).unsqueeze(0)  # (1, T)  → caller may add channel dim

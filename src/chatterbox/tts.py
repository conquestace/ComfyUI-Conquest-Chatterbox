from dataclasses import dataclass
from pathlib import Path
import librosa
import torch
import perth
import torch.nn.functional as F
import gc
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond
from .audio_editing import splice_audios


REPO_ID = "ResembleAI/chatterbox"


def punc_norm(text: str) -> str:
    if len(text) == 0:
        return "You need to add some text for me to talk."

    if text[0].islower():
        text = text[0].upper() + text[1:]

    text = " ".join(text.split())

    punc_to_replace = [
        ("...", ", "), ("…", ", "), (":", ","), (" - ", ", "), (";", ", "),
        ("—", "-"), ("–", "-"), (" ,", ","), ("“", "\""), ("”", "\""),
        ("‘", "'"), ("’", "'"),
    ]
    for old, new in punc_to_replace:
        text = text.replace(old, new)

    if not any(text.endswith(p) for p in {".", "!", "?", "-", ","}):
        text += "."
    return text


def chunk_text(text: str, chunk_size: int = 300):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


@dataclass
class Conditionals:
    t3: T3Cond
    gen: dict

    def to(self, device):
        # Move all tensors in the dataclass individually to avoid issues
        moved = {}
        for k, v in self.t3.__dict__.items():
            if torch.is_tensor(v):
                moved[k] = v.to(device)
            else:
                moved[k] = v
        self.t3 = T3Cond(**moved)

        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device)
        return self

    def save(self, fpath: Path):
        torch.save(dict(t3=self.t3.__dict__, gen=self.gen), fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(self, t3, s3gen, ve, tokenizer, device, conds=None):
        self.sr = S3GEN_SR
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTTS':
        ckpt_dir = Path(ckpt_dir)

        map_loc = "cpu" if device in {"cpu", "mps"} else None

        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors", map_location=map_loc))
        ve.to(device).eval()

        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors", map_location=map_loc)
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors", map_location=map_loc), strict=False)
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(str(ckpt_dir / "tokenizer.json"))

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTTS':
        for f in [
            "ve.safetensors",
            "t3_cfg.safetensors",
            "s3gen.safetensors",
            "tokenizer.json",
            "conds.pt",
        ]:
            hf_hub_download(repo_id=REPO_ID, filename=f)
        return cls.from_local(Path(hf_hub_download(repo_id=REPO_ID, filename="ve.safetensors")).parent, device)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        s3gen_ref_wav, _ = librosa.load(wav_fpath, sr=S3GEN_SR)
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[: self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        ve_embed = torch.from_numpy(
            self.ve.embeds_from_wavs([ref_16k_wav[: self.ENC_COND_LEN]], sample_rate=S3_SR)
        )
        ve_embed = ve_embed.mean(axis=0, keepdim=True)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        )
        self.conds = Conditionals(t3_cond, s3gen_ref_dict).to(self.device)

    def _generate_segment(
        self,
        text,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        top_p=0.8,
        repetition_penalty=2.0,
        min_p=0.0,
    ):
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration)
        else:
            assert self.conds is not None, "prepare_conditionals or audio_prompt_path required"

        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _c = self.conds.t3
            new_t3 = T3Cond(
                speaker_emb=_c.speaker_emb,
                cond_prompt_speech_tokens=_c.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            )
            moved = {}
            for k, v in new_t3.__dict__.items():
                moved[k] = v.to(self.device) if torch.is_tensor(v) else v
            self.conds.t3 = T3Cond(**moved)

        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                cfg_weight=cfg_weight,
            )[0]

            speech_tokens = drop_invalid_tokens(speech_tokens).to(self.device)
            wav, _ = self.s3gen.inference(speech_tokens=speech_tokens, ref_dict=self.conds.gen)
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)

    def generate(
        self,
        text,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        top_p=0.8,
        repetition_penalty=2.0,
        min_p=0.0,
        chunk_size: int = 300,
        max_retries: int = 1,
    ):
        def _safe_generate_segment(t):
            for _ in range(max_retries + 1):
                try:
                    return self._generate_segment(
                        t,
                        audio_prompt_path=audio_prompt_path,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        min_p=min_p,
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    raise

        if len(text) <= chunk_size:
            return _safe_generate_segment(text)

        segments = []
        for chunk in chunk_text(text, chunk_size):
            segment = _safe_generate_segment(chunk)
            segments.append(segment.squeeze(0).cpu().numpy())

        merged = splice_audios(segments)
        return torch.from_numpy(merged).unsqueeze(0)

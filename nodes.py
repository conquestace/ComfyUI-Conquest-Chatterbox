import os
import tempfile
import random
import numpy as np
import torch
import soundfile as sf

from chatterbox.tts import ChatterboxTTS
from chatterbox.vc  import ChatterboxVC
from chatterbox.audio_editing import (
    splice_audios,
    trim_audio,
    insert_audio,
    delete_segment,
    crossfade,
)

# -------------------------------------------------------------------------
# Helper utilities
# -------------------------------------------------------------------------

def _seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def _to_numpy_mono(audio_dict):
    """Convert ComfyUI AUDIO → (wav, sr) numpy mono float32."""
    wav = audio_dict["waveform"].cpu()
    sr  = audio_dict["sample_rate"]
    if wav.ndim == 3:            # (B,C,T)
        wav = wav[0]             #   -> (C,T)
    if wav.shape[0] > 1:         # stereo → mono
        wav = wav.mean(0)
    return wav.numpy().astype(np.float32), sr

def _numpy_to_comfy(wav_np: np.ndarray, sr: int):
    """Convert mono numpy wav → ComfyUI AUDIO dict."""
    tensor = torch.from_numpy(wav_np).unsqueeze(0).unsqueeze(0)  # (1,1,T)
    return {"waveform": tensor, "sample_rate": sr}

def _empty_audio(sr: int = 24_000):
    return _numpy_to_comfy(np.zeros(sr, dtype=np.float32), sr)

DEVICE_DEFAULT = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------------------
# 1) Text-to-Speech  (now with min_p, top_p, repetition_penalty)
# -------------------------------------------------------------------------

class ChatterboxTTSExtended:
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION     = "generate"
    CATEGORY     = "audio/generation"
    OUTPUT_NODE  = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required" : {
                "text"      : ("STRING",  {"multiline": True,
                                           "default": "Hello from ComfyUI 🤖"}),
                "exaggeration" : ("FLOAT", {"default": 0.5, "min": 0.25,
                                            "max": 2.0, "step": 0.05}),
                "temperature"  : ("FLOAT", {"default": 0.8, "min": 0.05,
                                            "max": 5.0, "step": 0.05}),
                "cfg_weight"   : ("FLOAT", {"default": 0.5, "min": 0.0,
                                            "max": 1.0, "step": 0.05}),
                "min_p"        : ("FLOAT", {"default": 0.05, "min": 0.0,
                                            "max": 1.0, "step": 0.01}),
                "top_p"        : ("FLOAT", {"default": 1.00, "min": 0.0,
                                            "max": 1.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 1.0,
                                                 "max": 2.0, "step": 0.1}),
                "seed"         : ("INT", {"default": 0, "min": 0,
                                          "max": 0xffffffffffffffff,
                                          "control_after_generate": True}),
                "device"       : (["cuda","cpu"], {"default": DEVICE_DEFAULT}),
            },
            "optional": {
                "audio_prompt": ("AUDIO",),   # reference voice
            }
        }

    # cache one model per device to avoid reloads
    _models = {}

    @classmethod
    def _get_model(cls, device):
        if device not in cls._models:
            cls._models[device] = ChatterboxTTS.from_pretrained(device)
        return cls._models[device]

    def generate(self, text, exaggeration, temperature, cfg_weight,
                 min_p, top_p, repetition_penalty, seed, device,
                 audio_prompt=None):

        if not text.strip():
            return (_empty_audio(),)

        if seed != 0:
            _seed_everything(int(seed))

        model = self._get_model(device)

        prompt_path = None
        try:
            if audio_prompt and audio_prompt["waveform"].numel() > 0:
                wav_np, sr = _to_numpy_mono(audio_prompt)
                fd, prompt_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                sf.write(prompt_path, wav_np, sr)

            wav = model.generate(
                text,
                audio_prompt_path=prompt_path,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight,
                min_p=min_p,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            return (_numpy_to_comfy(wav.squeeze(0).cpu().numpy(), model.sr),)
        finally:
            if prompt_path and os.path.exists(prompt_path):
                os.remove(prompt_path)

# -------------------------------------------------------------------------
# 2) Voice Conversion
# -------------------------------------------------------------------------

class ChatterboxVoiceConversion:
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("converted_audio",)
    FUNCTION     = "convert"
    CATEGORY     = "audio/generation"
    OUTPUT_NODE  = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_audio": ("AUDIO",),
                "device": (["cuda", "cpu"], {"default": DEVICE_DEFAULT}),
            },
            "optional": {
                "target_voice_audio": ("AUDIO",),
            }
        }

    _models = {}

    @classmethod
    def _get_model(cls, device):
        if device not in cls._models:
            cls._models[device] = ChatterboxVC.from_pretrained(device)
        return cls._models[device]

    def convert(self, source_audio, device, target_voice_audio=None):
        if source_audio is None or source_audio["waveform"].numel() == 0:
            return (_empty_audio(),)

        model = self._get_model(device)

        # save temp wavs -----------------------------------------------------
        src_path = tgt_path = None
        try:
            wav_src, sr_src = _to_numpy_mono(source_audio)
            fd, src_path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
            sf.write(src_path, wav_src, sr_src)

            if target_voice_audio and target_voice_audio["waveform"].numel() > 0:
                wav_tgt, sr_tgt = _to_numpy_mono(target_voice_audio)
                if sr_tgt != sr_src:
                    raise ValueError("Source and target voice must share SR")
                fd, tgt_path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
                sf.write(tgt_path, wav_tgt, sr_tgt)

            wav_out = model.generate(src_path, target_voice_path=tgt_path)
            return (_numpy_to_comfy(wav_out.squeeze(0).cpu().numpy(), model.sr),)
        finally:
            for p in (src_path, tgt_path):
                if p and os.path.exists(p):
                    os.remove(p)

# -------------------------------------------------------------------------
# 3) Audio-editing nodes  (splice, trim, insert, delete, cross-fade)
# -------------------------------------------------------------------------

class ChatterboxAudioSplice:
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION     = "splice"
    CATEGORY     = "audio/editing"
    OUTPUT_NODE  = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio_a": ("AUDIO",),
            "audio_b": ("AUDIO",),
        }}

    def splice(self, audio_a, audio_b):
        wav1, sr1 = _to_numpy_mono(audio_a)
        wav2, sr2 = _to_numpy_mono(audio_b)
        if sr1 != sr2:
            raise ValueError("Sampling rates must match")
        joined = splice_audios([wav1, wav2])
        return (_numpy_to_comfy(joined, sr1),)

class ChatterboxAudioTrim:
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION     = "trim"
    CATEGORY     = "audio/editing"
    OUTPUT_NODE  = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "start_sec": ("FLOAT", {"default": 0.0, "min": 0}),
            "end_sec"  : ("FLOAT", {"default": 1.0, "min": 0.01}),
        }}

    def trim(self, audio, start_sec, end_sec):
        wav, sr = _to_numpy_mono(audio)
        out = trim_audio(wav, float(start_sec), float(end_sec), sr)
        return (_numpy_to_comfy(out, sr),)

class ChatterboxAudioInsert:
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION     = "insert"
    CATEGORY     = "audio/editing"
    OUTPUT_NODE  = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "base_audio"  : ("AUDIO",),
            "insert_audio": ("AUDIO",),
            "position_sec": ("FLOAT", {"default": 0.0, "min": 0}),
        }}

    def insert(self, base_audio, insert_audio, position_sec):
        base_wav, sr = _to_numpy_mono(base_audio)
        ins_wav,  sr2 = _to_numpy_mono(insert_audio)
        if sr != sr2:
            raise ValueError("Sampling rates must match")
        out = insert_audio(base_wav, ins_wav, float(position_sec), sr)
        return (_numpy_to_comfy(out, sr),)

class ChatterboxAudioDeleteSegment:
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION     = "delete"
    CATEGORY     = "audio/editing"
    OUTPUT_NODE  = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "start_sec": ("FLOAT", {"default": 0.0, "min": 0}),
            "end_sec"  : ("FLOAT", {"default": 1.0, "min": 0.01}),
        }}

    def delete(self, audio, start_sec, end_sec):
        wav, sr = _to_numpy_mono(audio)
        out = delete_segment(wav, float(start_sec), float(end_sec), sr)
        return (_numpy_to_comfy(out, sr),)

class ChatterboxAudioCrossfade:
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION     = "crossfade"
    CATEGORY     = "audio/editing"
    OUTPUT_NODE  = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio_a": ("AUDIO",),
            "audio_b": ("AUDIO",),
            "duration_sec": ("FLOAT", {"default": 0.01, "min": 0.001}),
        }}

    def crossfade(self, audio_a, audio_b, duration_sec):
        wav1, sr1 = _to_numpy_mono(audio_a)
        wav2, sr2 = _to_numpy_mono(audio_b)
        if sr1 != sr2:
            raise ValueError("Sampling rates must match")
        out = crossfade(wav1, wav2, float(duration_sec), sr1)
        return (_numpy_to_comfy(out, sr1),)

# -------------------------------------------------------------------------
# Register with ComfyUI
# -------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "Chatterbox TTS (extended)"     : ChatterboxTTSExtended,
    "Chatterbox Voice Conversion"   : ChatterboxVoiceConversion,
    "Chatterbox Splice Audio"       : ChatterboxAudioSplice,
    "Chatterbox Trim Audio"         : ChatterboxAudioTrim,
    "Chatterbox Insert Audio"       : ChatterboxAudioInsert,
    "Chatterbox Delete Segment"     : ChatterboxAudioDeleteSegment,
    "Chatterbox Cross-fade Audio"   : ChatterboxAudioCrossfade,
}

__all__ = ["NODE_CLASS_MAPPINGS"]

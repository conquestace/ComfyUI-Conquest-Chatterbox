import numpy as np


def splice_audios(segments, xfade_ms=40, sr=24000):
    """Concatenate segments with a short cross‑fade to avoid pops."""
    if not segments:
        return np.array([], dtype=np.float32)

    xfade = int(sr * xfade_ms / 1000)
    output = np.array(segments[0], dtype=np.float32)
    for seg in segments[1:]:
        seg = np.array(seg, dtype=np.float32)
        if xfade > 0 and len(output) > xfade and len(seg) > xfade:
            fade_out = np.linspace(1.0, 0.0, xfade, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, xfade, dtype=np.float32)
            cross = output[-xfade:] * fade_out + seg[:xfade] * fade_in
            output = np.concatenate([output[:-xfade], cross, seg[xfade:]])
        else:
            output = np.concatenate([output, seg])
    return output

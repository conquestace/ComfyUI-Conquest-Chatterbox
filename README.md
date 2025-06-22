<div align="center">

<h1>conquest-chatterbox</h1>

<p align="center">
  <img src="./assets/preview.png" alt="ComfyUI-KEEP Workflow Example">
</p>
    
</div>

Custom nodes for ComfyUI that integrate the [Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox) library for Text-to-Speech (TTS) and Voice Conversion (VC).

## 📢 Features

*   **conquest-chatterbox Node:**
    *   Synthesize speech from text.
    *   Optional voice cloning using an audio prompt.
    *   Adjustable parameters: exaggeration, temperature, CFG weight, seed.
    *   Handles long passages by automatically splitting text into manageable chunks.
*   **conquest-chatterbox Node:**
    *   Convert the voice in a source audio file to sound like a target voice.
    *   Uses a target audio file for voice characteristics or defaults to a built-in voice if no target is provided.
*   **Automatic Model Downloading:** Necessary model files are automatically downloaded from Hugging Face (`ResembleAI/chatterbox`) on first use if not found locally.


## 🎭 conquest-chatterbox Demo Samples
Check the [official demo](https://resemble-ai.github.io/chatterbox_demopage/)

## Installation
Via ComfyUI Manager [conquest-chatterbox](https://registry.comfy.org/publishers/wildai/nodes/ComfyUI-ChatterboxTTS) or by cloning the repo:

1.  **Clone this repository:**
    ```bash
    git clone https://github.com/wildminder/ComfyUI-Chatterbox.git ComfyUI/custom_nodes/ComfyUI-Chatterbox
    ```

2.  **Install Dependencies:**
    Navigate to the custom node's directory and install the required packages:
    ```bash
    cd ComfyUI/custom_nodes/ComfyUI-Chatterbox
    pip install -r requirements.txt
    ```

3.  **Model Pack Directory (Automatic Setup):**
    The node will automatically attempt to download the default model pack (`resembleai_default_voice`) into `ComfyUI/models/chatterbox_tts/` when you first use a node that requires it.
    You can also manually create subdirectories in `ComfyUI/models/chatterbox_tts/` and place other Chatterbox model packs there. Each pack should contain:
    *   `ve.pt`
    *   `t3_cfg.pt`
    *   `s3gen.pt`
    *   `tokenizer.json`
    *   `conds.pt` (for default voice capabilities)

4.  **Restart ComfyUI.**

## Usage

After installation and restarting ComfyUI:

*   The **"conquest-chatterbox"** node will be available under the `audio/generation` and `audio/conversion` categories.

Load example workflows from the `workflow-examples/` directory in this repository to get started.

## Notes

*   The Chatterbox library is included within this custom node's `src/` directory.
*   Tested with Pytorch 2.7 + CUDA 12.6

## Acknowledgements

*   This node relies on the [Chatterbox](https://github.com/resemble-ai/chatterbox) library by Resemble AI.

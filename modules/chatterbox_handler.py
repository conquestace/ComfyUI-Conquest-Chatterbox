import os
import sys
import torch
import random
import numpy as np
import folder_paths
import hashlib
from huggingface_hub import hf_hub_download

current_script_path = os.path.dirname(os.path.abspath(__file__))
chatterbox_src_path = os.path.abspath(os.path.join(current_script_path, '..', 'src'))
if chatterbox_src_path not in sys.path:
    sys.path.insert(0, chatterbox_src_path)


from chatterbox.tts import ChatterboxTTS
from chatterbox.vc import ChatterboxVC # <--- Added VC import

TTS_MODEL_CACHE = {}
VC_MODEL_CACHE = {}
CHATTERBOX_MODEL_SUBDIR = "chatterbox_tts"
CHATTERBOX_REPO_ID = "ResembleAI/chatterbox"

CHATTERBOX_FILES_TO_DOWNLOAD = ["ve.pt", "t3_cfg.pt", "s3gen.pt", "tokenizer.json", "conds.pt"]
DEFAULT_MODEL_PACK_NAME = "resembleai_default_voice"

def get_chatterbox_model_pack_names():
    """Returns a list of available Chatterbox model pack names (subdirectories)."""
    chatterbox_models_base_path = os.path.join(folder_paths.models_dir, CHATTERBOX_MODEL_SUBDIR)
    if not os.path.isdir(chatterbox_models_base_path):
        os.makedirs(chatterbox_models_base_path, exist_ok=True)
        #print(f"ChatterboxTTS/VC: Created models directory at {chatterbox_models_base_path}")
        return []
    
    packs = [d for d in os.listdir(chatterbox_models_base_path) if os.path.isdir(os.path.join(chatterbox_models_base_path, d))]
    if not packs:
        return []
    return packs

def get_model_pack_path(model_pack_name):
    """Gets the full path to a specific model pack."""
    if not model_pack_name:
        return None
    return os.path.join(folder_paths.models_dir, CHATTERBOX_MODEL_SUBDIR, model_pack_name)

def _download_file_from_hf(repo_id, filename, local_dir, local_filename=None):
    if local_filename is None:
        local_filename = filename
    destination = os.path.join(local_dir, local_filename)
    
    if not os.path.exists(destination):
        print(f"ChatterboxTTS/VC: Downloading '{filename}' from '{repo_id}' to '{destination}'...")
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
            )
            if filename != local_filename and os.path.exists(os.path.join(local_dir, filename)) and not os.path.exists(destination):
                 os.rename(os.path.join(local_dir, filename), destination)
            print(f"ChatterboxTTS/VC: Successfully downloaded '{local_filename}'.")
            return True
        except Exception as e:
            print(f"ChatterboxTTS/VC: Failed to download '{filename}' from '{repo_id}'. Error: {e}")
            if os.path.exists(destination + ".incomplete"):
                os.remove(destination + ".incomplete")
            return False
    else:
        return True

def download_chatterbox_model_pack_if_missing(model_pack_name):
    """Downloads all necessary model files for a given pack name if they don't exist."""
    ckpt_dir = get_model_pack_path(model_pack_name)
    if not ckpt_dir:
        print(f"ChatterboxTTS/VC: Invalid model pack name '{model_pack_name}', cannot download.")
        return False
        
    os.makedirs(ckpt_dir, exist_ok=True)
    
    all_files_successfully_managed = True

    vc_required_files = ["s3gen.pt", "conds.pt"]

    files_to_check = CHATTERBOX_FILES_TO_DOWNLOAD
    
    for file_name in files_to_check:
        if not _download_file_from_hf(CHATTERBOX_REPO_ID, file_name, ckpt_dir):
            all_files_successfully_managed = False
            if file_name in vc_required_files:
                print(f"ChatterboxTTS/VC: Critical file '{file_name}' for VC failed to download for pack '{model_pack_name}'.")
            # return False

    if all_files_successfully_managed:
        pass #print(f"ChatterboxTTS/VC: Verified/Downloaded all files for model pack '{model_pack_name}' into '{ckpt_dir}'.")
    else:
        print(f"ChatterboxTTS/VC: Some files failed to download for model pack '{model_pack_name}'. Check logs.")
    return all_files_successfully_managed


def load_chatterbox_tts_model(model_pack_name, device_str="cuda"):
    ckpt_dir = get_model_pack_path(model_pack_name)
    if not ckpt_dir:
        raise ValueError(f"ChatterboxTTS: Invalid model_pack_name: {model_pack_name}")
    #print(f"ChatterboxTTS: Attempting to load TTS model pack '{model_pack_name}' from '{ckpt_dir}'.")
    if not download_chatterbox_model_pack_if_missing(model_pack_name):
        print(f"ChatterboxTTS: Warning - Not all TTS model files could be verified/downloaded for '{model_pack_name}'. Loading may fail.")
    if not os.path.isdir(ckpt_dir):
         raise FileNotFoundError(f"ChatterboxTTS: Model pack directory '{model_pack_name}' not found and could not be created at '{ckpt_dir}'.")
    #print(f"ChatterboxTTS: Loading ChatterboxTTS model from local directory: {ckpt_dir} onto device: {device_str}")
    try:
        model = ChatterboxTTS.from_local(ckpt_dir, device=device_str)
    except Exception as e:
        print(f"ChatterboxTTS: Error during ChatterboxTTS.from_local('{ckpt_dir}', device='{device_str}'): {e}")
        print(f"ChatterboxTTS: Please ensure all required model files ({', '.join(CHATTERBOX_FILES_TO_DOWNLOAD)}) are present in {ckpt_dir} or can be downloaded from {CHATTERBOX_REPO_ID}.")
        raise
    return model

def get_cached_chatterbox_tts_model(model_pack_name, device_str="cuda"):
    """Loads and caches the ChatterboxTTS model."""
    if not model_pack_name:
        available_packs = get_chatterbox_model_pack_names()
        model_pack_name = available_packs[0] if available_packs else DEFAULT_MODEL_PACK_NAME
        print(f"ChatterboxTTS: No model pack specified for TTS, using '{model_pack_name}'.")

    current_entry = TTS_MODEL_CACHE.get(device_str)
    if current_entry and current_entry[0] == model_pack_name:
        return current_entry[1]

    if current_entry:
        # Replace cached model for this device
        TTS_MODEL_CACHE.pop(device_str, None)

    print(f"ChatterboxTTS: Loading model '{model_pack_name}' on {device_str}...")
    model = load_chatterbox_tts_model(model_pack_name, device_str)
    TTS_MODEL_CACHE[device_str] = (model_pack_name, model)
    return model


def load_chatterbox_vc_model(model_pack_name, device_str="cuda"):
    """Loads the ChatterboxVC model from a specified pack."""
    ckpt_dir = get_model_pack_path(model_pack_name)
    if not ckpt_dir:
        raise ValueError(f"ChatterboxVC: Invalid model_pack_name: {model_pack_name}")

    #print(f"ChatterboxVC: Attempting to load VC model pack '{model_pack_name}' from '{ckpt_dir}'.")

    if not download_chatterbox_model_pack_if_missing(model_pack_name):
        vc_min_files = ["s3gen.pt", "conds.pt"]
        for f_name in vc_min_files:
            if not os.path.exists(os.path.join(ckpt_dir, f_name)):
                 print(f"ChatterboxVC: Critical file '{f_name}' for VC is missing from pack '{model_pack_name}' after download attempt. Loading will likely fail.")
                 break 
        else:
             print(f"ChatterboxVC: Warning - Not all model files could be verified/downloaded for '{model_pack_name}', but attempting to load VC with available files.")


    if not os.path.isdir(ckpt_dir):
         raise FileNotFoundError(f"ChatterboxVC: Model pack directory '{model_pack_name}' not found and could not be created at '{ckpt_dir}'.")
    
    # print(f"ChatterboxVC: Loading ChatterboxVC model from local directory: {ckpt_dir} onto device: {device_str}")
    try:
        model = ChatterboxVC.from_local(ckpt_dir, device=device_str)
    except Exception as e:
        print(f"ChatterboxVC: Error during ChatterboxVC.from_local('{ckpt_dir}', device='{device_str}'): {e}")
        print(f"ChatterboxVC: Please ensure at least 's3gen.pt' and optionally 'conds.pt' (for default voice) are present in {ckpt_dir} or can be downloaded from {CHATTERBOX_REPO_ID}.")
        raise
    return model

def get_cached_chatterbox_vc_model(model_pack_name, device_str="cuda"):
    """Loads and caches the ChatterboxVC model."""
    if not model_pack_name:
        available_packs = get_chatterbox_model_pack_names()
        model_pack_name = available_packs[0] if available_packs else DEFAULT_MODEL_PACK_NAME
        print(f"ChatterboxVC: No model pack specified for VC, using '{model_pack_name}'.")

    current_entry = VC_MODEL_CACHE.get(device_str)
    if current_entry and current_entry[0] == model_pack_name:
        return current_entry[1]

    if current_entry:
        VC_MODEL_CACHE.pop(device_str, None)

    print(f"ChatterboxVC: Loading model '{model_pack_name}' on {device_str}...")
    model = load_chatterbox_vc_model(model_pack_name, device_str)
    VC_MODEL_CACHE[device_str] = (model_pack_name, model)
    return model


def set_chatterbox_seed(seed: int):
    """Sets the seed for Chatterbox TTS/VC."""
    MAX_NUMPY_SEED = 2**32 - 1
    if seed == 0: 
        actual_seed_for_torch_random = random.randint(1, 0xffffffffffffffff)
        actual_seed_for_numpy = random.randint(1, MAX_NUMPY_SEED)
    else:
        actual_seed_for_torch_random = seed
        # fast fix range for seed, sorry 
        actual_seed_for_numpy = seed % MAX_NUMPY_SEED
    
    torch.manual_seed(actual_seed_for_torch_random)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed_for_torch_random)
    random.seed(actual_seed_for_torch_random)
    np.random.seed(actual_seed_for_numpy)
    
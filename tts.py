import asyncio
import datetime
import logging
import os
import time
import traceback

import edge_tts
import gradio as gr
import librosa
import torch
from fairseq import checkpoint_utils

from config import Config
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from rmvpe import RMVPE
from vc_infer_pipeline import VC

logging.getLogger("fairseq").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

limitation = os.getenv("SYSTEM") == "spaces"

config = Config()

edge_output_filename = "edge_output.mp3"
tts_voice_list = asyncio.get_event_loop().run_until_complete(edge_tts.list_voices())
tts_voices = [f"{v['ShortName']}-{v['Gender']}" for v in tts_voice_list]

model_root = "weights"
models = [
    d for d in os.listdir(model_root) if os.path.isdir(os.path.join(model_root, d))
]
if len(models) == 0:
    raise ValueError("No model found in `weights` folder")
models.sort()


def model_data(model_name):
    # global n_spk, tgt_sr, net_g, vc, cpt, version, index_file
    pth_files = [
        os.path.join(model_root, model_name, f)
        for f in os.listdir(os.path.join(model_root, model_name))
        if f.endswith(".pth")
    ]
    if len(pth_files) == 0:
        raise ValueError(f"No pth file found in {model_root}/{model_name}")
    pth_path = pth_files[0]
    print(f"Loading {pth_path}")
    cpt = torch.load(pth_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    else:
        raise ValueError("Unknown version")
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    print("Model loaded")
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    # n_spk = cpt["config"][-3]

    index_files = [
        os.path.join(model_root, model_name, f)
        for f in os.listdir(os.path.join(model_root, model_name))
        if f.endswith(".index")
    ]
    if len(index_files) == 0:
        print("No index file found")
        index_file = ""
    else:
        index_file = index_files[0]
        print(f"Index file found: {index_file}")

    return tgt_sr, net_g, vc, version, index_file, if_f0


def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()


print("Loading hubert model...")
hubert_model = load_hubert()
print("Hubert model loaded.")

print("Loading rmvpe model...")
rmvpe_model = RMVPE("rmvpe.pt", config.is_half, config.device)
print("rmvpe model loaded.")


def tts(
    model_name,
    speed,
    tts_text,
    tts_voice,
    f0_up_key,
    f0_method,
    index_rate,
    protect,
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=0.25,
):
    print("------------------")
    print(datetime.datetime.now())
    print("tts_text:")
    print(tts_text)
    print(f"tts_voice: {tts_voice}")
    print(f"Model name: {model_name}")
    print(f"F0: {f0_method}, Key: {f0_up_key}, Index: {index_rate}, Protect: {protect}")
    try:
        if limitation and len(tts_text) > 280:
            print("Error: Text too long")
            return (
                f"Text characters should be at most 280 in this huggingface space, but got {len(tts_text)} characters.",
                None,
                None,
            )
        tgt_sr, net_g, vc, version, index_file, if_f0 = model_data(model_name)
        t0 = time.time()
        if speed >= 0:
            speed_str = f"+{speed}%"
        else:
            speed_str = f"{speed}%"
        asyncio.run(
            edge_tts.Communicate(
                tts_text, "-".join(tts_voice.split("-")[:-1]), rate=speed_str
            ).save(edge_output_filename)
        )
        t1 = time.time()
        edge_time = t1 - t0
        audio, sr = librosa.load(edge_output_filename, sr=16000, mono=True)
        duration = len(audio) / sr
        print(f"Audio duration: {duration}s")
        if limitation and duration >= 20:
            print("Error: Audio too long")
            return (
                f"Audio should be less than 20 seconds in this huggingface space, but got {duration}s.",
                edge_output_filename,
                None,
            )

        f0_up_key = int(f0_up_key)

        if not hubert_model:
            load_hubert()
        if f0_method == "rmvpe":
            vc.model_rmvpe = rmvpe_model
        times = [0, 0, 0]
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            0,
            audio,
            edge_output_filename,
            times,
            f0_up_key,
            f0_method,
            index_file,
            # file_big_npy,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            None,
        )
        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr
        info = f"Success. Time: edge-tts: {edge_time}s, npy: {times[0]}s, f0: {times[1]}s, infer: {times[2]}s"
        print(info)
        return (
            info,
            edge_output_filename,
            (tgt_sr, audio_opt),
        )
    except EOFError:
        info = (
            "It seems that the edge-tts output is not valid. "
            "This may occur when the input text and the speaker do not match. "
            "For example, maybe you entered Japanese (without alphabets) text but chose non-Japanese speaker?"
        )
        print(info)
        return info, None, None
    except:
        info = traceback.format_exc()
        print(info)
        return info, None, None

"""Copyright: Nabarun Goswami (2024)."""
from tts_asr_eval_suite.secs.unispeech_ecapa_tdnn.ecapa_tdnn import ECAPA_TDNN_SMALL

import torch
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download


class SECSWavLMLargeSV:
    def __init__(self, device) -> None:
        self.device = device
        self.sr = 16000

        s3prl_ckpt = hf_hub_download(repo_id="subatomicseer/wavlm-large-sv-ckpts", filename="wavlm_large.pt", cache_dir=None)
        wavlm_large_finetune_ckpt = hf_hub_download(repo_id="subatomicseer/wavlm-large-sv-ckpts", filename="wavlm_large_finetune.pth", cache_dir=None)

        self.sv_model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=s3prl_ckpt)
        state_dict = torch.load(wavlm_large_finetune_ckpt, map_location=lambda storage, loc: storage)
        self.sv_model.load_state_dict(state_dict['model'], strict=False)
        self.sv_model = self.sv_model.eval().to(device)

    def __call__(self, prompt_path, gen_path):

        prompt_audio, sr_prompt = torchaudio.load(prompt_path)
        gen_audio, sr_gen = torchaudio.load(gen_path)

        prompt_audio = prompt_audio.to(self.device)
        gen_audio = gen_audio.to(self.device)

        if sr_prompt != self.sr:
            prompt_audio = torchaudio.functional.resample(prompt_audio, sr_prompt, self.sr)

        if sr_gen != self.sr:
            gen_audio = torchaudio.functional.resample(gen_audio, sr_gen, self.sr)

        prompt_embedding = self.sv_model(prompt_audio)
        gen_embedding = self.sv_model(gen_audio)

        similarity = F.cosine_similarity(prompt_embedding, gen_embedding)[0].item()

        return similarity

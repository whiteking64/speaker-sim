"""Copyright: Nabarun Goswami (2024)."""
from tts_asr_eval_suite.unispeech_ecapa_tdnn.ecapa_tdnn import ECAPA_TDNN_SMALL

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


class SECSWavLMLargeSV:
    def __init__(self, device) -> None:
        self.device = device

        s3prl_ckpt = hf_hub_download(repo_id="subatomicseer/wavlm-large-sv-ckpts", filename="wavlm_large.pt", cache_dir=None)
        wavlm_large_finetune_ckpt = hf_hub_download(repo_id="subatomicseer/wavlm-large-sv-ckpts", filename="wavlm_large_finetune.pth", cache_dir=None)

        self.sv_model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=s3prl_ckpt)
        state_dict = torch.load(wavlm_large_finetune_ckpt, map_location=lambda storage, loc: storage)
        self.sv_model.load_state_dict(state_dict['model'], strict=False)
        self.sv_model = self.sv_model.eval().to(device)

    def __call__(self, prompt_audio, gen_audio):

        prompt_embedding = self.sv_model(prompt_audio)
        gen_embedding = self.sv_model(gen_audio)

        similarity = F.cosine_similarity(prompt_embedding, gen_embedding).squeeze().item()

        return similarity

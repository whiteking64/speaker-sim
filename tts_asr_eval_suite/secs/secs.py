import os
import gdown
import torch
import torchaudio
import torch.nn.functional as F

from tts_asr_eval_suite.secs.ecapa_tdnn import ECAPA_TDNN_SMALL

WAVLM_LARGE_URL = 'https://drive.google.com/uc?id=1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP'

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(THIS_DIR, 'ckpt')
WAVLM_LARGE_PATH = os.path.join(CKPT_DIR, 'wavlm_large_finetune.pth')


class SECS:
    def __init__(self, device) -> None:
        self.device = device

        self.sv_model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)

        if not os.path.exists(WAVLM_LARGE_PATH):
            os.makedirs(CKPT_DIR, exist_ok=True)
            gdown.download(WAVLM_LARGE_URL, WAVLM_LARGE_PATH, quiet=False)

        state_dict = torch.load(WAVLM_LARGE_PATH, map_location=lambda storage, loc: storage)
        self.sv_model.load_state_dict(state_dict["model"], strict=False)
        self.sv_model = self.sv_model.eval().to(device)

    def __call__(self, audio1path, audio2path):

        audio_1, sr1 = torchaudio.load(audio1path)
        audio_2, sr2 = torchaudio.load(audio2path)

        audio_1 = audio_1.to(self.device)
        audio_2 = audio_2.to(self.device)

        if sr1 != self.sv_model.sr:
            audio_1 = torchaudio.functional.resample(audio_1, sr1, self.sv_model.sr)

        if sr2 != self.sv_model.sr:
            audio_2 = torchaudio.functional.resample(audio_2, sr2, self.sv_model.sr)

        with torch.inference_mode():
            embedding1 = self.sv_model(audio_1)
            embedding2 = self.sv_model(audio_2)

        similarity = F.cosine_similarity(embedding1, embedding2)[0].item()

        return similarity

import torch
import torchaudio
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


class SECS:
    def __init__(self, device) -> None:
        self.device = device
        self.sr = 16000

        model_file = hf_hub_download(repo_id='Jenthe/ECAPA2', filename='ecapa2.pt', cache_dir=None)

        self.sv_model = torch.jit.load(model_file, map_location=device)
        self.sv_model.half()

    def __call__(self, audio1path, audio2path):

        audio_1, sr1 = torchaudio.load(audio1path)
        audio_2, sr2 = torchaudio.load(audio2path)

        audio_1 = audio_1.to(self.device)
        audio_2 = audio_2.to(self.device)

        if sr1 != self.sr:
            audio_1 = torchaudio.functional.resample(audio_1, sr1, self.sr)

        if sr2 != self.sr:
            audio_2 = torchaudio.functional.resample(audio_2, sr2, self.sr)

        embedding1 = self.sv_model(audio_1)
        embedding2 = self.sv_model(audio_2)

        similarity = F.cosine_similarity(embedding1, embedding2)[0].item()

        return similarity

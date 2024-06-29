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

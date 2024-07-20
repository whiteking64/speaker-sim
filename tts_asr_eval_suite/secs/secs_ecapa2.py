import torch
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download


class SECSEcapa2:
    def __init__(self, device) -> None:
        self.device = device
        model_file = hf_hub_download(repo_id='Jenthe/ECAPA2', filename='ecapa2.pt', cache_dir=None)
        self.sv_model = torch.jit.load(model_file, map_location=device)
        self.sv_model.half()

    def __call__(self, prompt_audio, gen_audio):

        prompt_embedding = self.sv_model(prompt_audio).float()
        gen_embedding = self.sv_model(gen_audio).float()

        similarity = F.cosine_similarity(prompt_embedding, gen_embedding).squeeze().item()

        return similarity

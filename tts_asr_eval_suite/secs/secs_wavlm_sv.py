import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector


class SECSWavLM:
    def __init__(self, device) -> None:
        self.device = device
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
        self.sv_model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv').to(device)

    def __call__(self, prompt_path, gen_path):
        prompt_audio, sr_prompt = torchaudio.load(prompt_path)
        gen_audio, sr_gen = torchaudio.load(gen_path)

        prompt_inputs = self.feature_extractor(prompt_audio.squeeze(), padding=True, return_tensors="pt")
        gen_inputs = self.feature_extractor(gen_audio.squeeze(), padding=True, return_tensors="pt")

        for k, v in prompt_inputs.items():
            prompt_inputs[k] = v.to(self.device)
        for k, v in gen_inputs.items():
            gen_inputs[k] = v.to(self.device)

        prompt_embeddings = self.sv_model(**prompt_inputs).embeddings
        gen_embeddings = self.sv_model(**gen_inputs).embeddings

        prompt_embeddings = F.normalize(prompt_embeddings, dim=-1)
        gen_embeddings = F.normalize(gen_embeddings, dim=-1)

        similarity = F.cosine_similarity(prompt_embeddings, gen_embeddings)[0].item()

        return similarity

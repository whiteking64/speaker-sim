import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector


class SECSWavLMBasePlusSV:
    def __init__(self, device) -> None:
        self.device = device
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
        self.sv_model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv').to(device)

    def __call__(self, prompt_audio, gen_audio):

        prompt_embeddings = self.sv_model(input_values=prompt_audio).embeddings
        gen_embeddings = self.sv_model(input_values=gen_audio).embeddings

        similarity = F.cosine_similarity(prompt_embeddings, gen_embeddings, dim=-1).squeeze().item()

        return similarity

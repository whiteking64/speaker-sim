import torch
import torchaudio

from resemblyzer import preprocess_wav

from tts_asr_eval_suite.secs_ecapa2 import SECSEcapa2
from tts_asr_eval_suite.secs_resemblyzer import SECSResemblyzer
from tts_asr_eval_suite.secs_unispeech_ecapa_wavlm_large import SECSWavLMLargeSV
from tts_asr_eval_suite.secs_wavlm_base_plus_sv import SECSWavLMBasePlusSV


def load_and_preprocess(file_path: str, target_sr: int, device: str) -> torch.Tensor:
    """Load and preprocess audio file."""
    audio, sr = torchaudio.load(file_path)

    # Convert stereo to mono if necessary
    if audio.shape[0] == 2:
        audio = audio.mean(dim=0, keepdim=True)

    # Resample if necessary
    if sr != target_sr:
        audio = torchaudio.transforms.Resample(sr, target_sr)(audio)

    audio = preprocess_wav(audio.squeeze().numpy(), source_sr=target_sr)
    audio = torch.from_numpy(audio).to(device).unsqueeze(0)

    return audio


class SECS:
    def __init__(self, device, methods) -> None:
        self.device = device
        self.sr = 16000
        if len(methods) == 0:
            methods = ['resemblyzer', 'wavlm_large_sv', 'wavlm_base_plus_sv', 'ecapa2']
        elif len(methods) == 1 and methods[0] == 'all':
            methods = ['resemblyzer', 'wavlm_large_sv', 'wavlm_base_plus_sv', 'ecapa2']

        self.scorers = {}
        for method in methods:
            if method == 'resemblyzer':
                self.scorers['resemblyzer'] = SECSResemblyzer(device)
            elif method == 'wavlm_large_sv':
                self.scorers['wavlm_large_sv'] = SECSWavLMLargeSV(device)
            elif method == 'wavlm_base_plus_sv':
                self.scorers['wavlm_base_plus_sv'] = SECSWavLMBasePlusSV(device)
            elif method == 'ecapa2':
                self.scorers['ecapa2'] = SECSEcapa2(device)
            else:
                raise ValueError(f"Invalid method: {method}")

    def __call__(self, prompt_path, gen_path):
        similarity = {}

        prompt_audio = load_and_preprocess(prompt_path, self.sr, self.device)
        gen_audio = load_and_preprocess(gen_path, self.sr, self.device)

        for method, scorer in self.scorers.items():
            similarity[f"SECS ({method})"] = scorer(prompt_audio, gen_audio)

        # similarity["SECS (avg)"] = sum([similarity[f"SECS ({method})"] for method in self.scorers]) / len(self.scorers)
        return similarity

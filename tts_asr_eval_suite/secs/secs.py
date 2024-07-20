import torch
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download

from tts_asr_eval_suite.secs.secs_ecapa2 import SECSEcapa2
from tts_asr_eval_suite.secs.secs_resemblyzer import SECSResemblyzer
from tts_asr_eval_suite.secs.secs_unispeech_ecapa_wavlm_large import SECSWavLMLargeSV
from tts_asr_eval_suite.secs.secs_wavlm_base_plus_sv import SECSWavLMBasePlusSV


class SECS:
    def __init__(self, device, method) -> None:
        self.device = device
        self.sr = 16000

        if method == 'resemblyzer':
            self.scorers = {
                'resemblyzer': SECSResemblyzer(device),
            }
        elif method == 'wavlm_large_sv':
            self.scorers = {
                'wavlm_large_sv': SECSWavLMLargeSV(device),
            }
        elif method == 'wavlm_base_plus_sv':
            self.scorers = {
                'wavlm_base_plus_sv': SECSWavLMBasePlusSV(device),
            }
        elif method == 'ecapa2':
            self.scorers = {
                'ecapa2': SECSEcapa2(device),
            }
        elif method == 'all':
            self.scorers = {
                'resemblyzer': SECSResemblyzer(device),
                'wavlm_large_sv': SECSWavLMLargeSV(device),
                'wavlm_base_plus_sv': SECSWavLMBasePlusSV(device),
                'ecapa2': SECSEcapa2(device),
            }
        else:
            raise ValueError(f"Invalid method: {method}")

    def __call__(self, prompt_path, gen_path):
        similarity = {}
        prompt_audio, sr_prompt = torchaudio.load(prompt_path)
        gen_audio, sr_gen = torchaudio.load(gen_path)

        prompt_audio = prompt_audio.to(self.device)
        gen_audio = gen_audio.to(self.device)

        if sr_prompt != self.sr:
            prompt_audio = torchaudio.functional.resample(prompt_audio, sr_prompt, self.sr)

        if sr_gen != self.sr:
            gen_audio = torchaudio.functional.resample(gen_audio, sr_gen, self.sr)

        for method, scorer in self.scorers.items():
            similarity[f"SECS ({method})"] = scorer(prompt_audio, gen_audio)

        similarity[f"SECS (avg)"] = sum([similarity[f"SECS ({method})"] for method in self.scorers]) / len(self.scorers)
        return similarity

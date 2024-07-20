import os
import requests
import torch
import torchaudio
from contextlib import redirect_stdout
import io
from torchaudio.pipelines import SQUIM_OBJECTIVE


class SQUIM:
    def __init__(self, device) -> None:
        self.device = device
        self.sr = 16000

        self.objective_model = SQUIM_OBJECTIVE.get_model().to(device)

    def __call__(self, gen_audio_path):

        gen_audio, gen_sr = torchaudio.load(gen_audio_path)

        gen_audio = gen_audio.to(self.device)

        if gen_sr != self.sr:
            gen_audio = torchaudio.functional.resample(gen_audio, gen_sr, self.sr)

        stoi_hyp, pesq_hyp, si_sdr_hyp = self.objective_model(gen_audio[0:1, :])

        return {"STOI (SQUIM)": stoi_hyp.item(),
                "PESQ (SQUIM)": pesq_hyp.item(),
                "SI-SDR (SQUIM)": si_sdr_hyp.item()}

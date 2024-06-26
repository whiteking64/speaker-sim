import os
import requests
import discrete_speech_metrics as dsm
import torch
import torchaudio


class UTMOS:
    def __init__(self, device) -> None:
        self.device = device
        self.sr = 16000

        # This is fix because the hub.load throws error and downloads incomplete file
        url = "https://github.com/tarepan/SpeechMOS/releases/download/v1.0.0/utmos22_strong_step7459_v1.pt"
        local_path = os.path.join(torch.hub.get_dir(), 'checkpoints', 'utmos22_strong_step7459_v1.pt')
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, 'wb') as f:
            f.write(requests.get(url).content)

        self.metric = dsm.UTMOS(sr=16000)

    def __call__(self, gt_audio_path, gen_audio_path):

        gt_audio, gt_sr = torchaudio.load(gt_audio_path)
        gen_audio, gen_sr = torchaudio.load(gen_audio_path)

        if gt_sr != self.sr:
            gt_audio = torchaudio.functional.resample(gt_audio, gt_sr, self.sr)
        if gen_sr != self.sr:
            gen_audio = torchaudio.functional.resample(gen_audio, gen_sr, self.sr)

        gt_audio = gt_audio[0].numpy()
        gen_audio = gen_audio[0].numpy()

        assert gt_audio.shape[0] == gen_audio.shape[0], \
            (f"Audio files have different lengths: {gt_audio.shape[0]} != {gen_audio.shape[0]}, "
             f"PESQ requires audio files to have the same length.")

        score = self.metric.score(gen_audio)

        return score

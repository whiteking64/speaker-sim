import discrete_speech_metrics as dsm
import torchaudio


class MelCepstralDistortion:
    def __init__(self, device) -> None:
        self.device = device
        self.sr = 16000
        self.metric = dsm.MCD(sr=16000)

    def __call__(self, gt_audio_path, gen_audio_path):

        gt_audio, gt_sr = torchaudio.load(gt_audio_path)
        gen_audio, gen_sr = torchaudio.load(gen_audio_path)

        if gt_sr != self.sr:
            gt_audio = torchaudio.functional.resample(gt_audio, gt_sr, self.sr)
        if gen_sr != self.sr:
            gen_audio = torchaudio.functional.resample(gen_audio, gen_sr, self.sr)

        gt_audio = gt_audio[0].numpy()
        gen_audio = gen_audio[0].numpy()

        score = self.metric.score(gt_audio, gen_audio)

        return score

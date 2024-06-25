import discrete_speech_metrics as dsm
import torchaudio


class UTMOS:
    def __init__(self, device) -> None:
        self.device = device
        self.sr = 16000
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

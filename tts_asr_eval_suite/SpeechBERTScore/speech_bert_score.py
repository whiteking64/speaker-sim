
import torchaudio
from contextlib import redirect_stdout
import io
stdout_capture = io.StringIO()
with redirect_stdout(stdout_capture):
    import discrete_speech_metrics as dsm


class SpeechBERTScore:
    def __init__(self, device) -> None:
        self.device = device
        self.sr = 16000
        self.metric = dsm.SpeechBERTScore(
            sr=16000,
            model_type="wavlm-large",
            layer=14,
            use_gpu=True)

    def __call__(self, gt_audio_path, gen_audio_path):

        gt_audio, gt_sr = torchaudio.load(gt_audio_path)
        gen_audio, gen_sr = torchaudio.load(gen_audio_path)

        if gt_sr != self.sr:
            gt_audio = torchaudio.functional.resample(gt_audio, gt_sr, self.sr)
        if gen_sr != self.sr:
            gen_audio = torchaudio.functional.resample(gen_audio, gen_sr, self.sr)

        gt_audio = gt_audio[0].numpy()
        gen_audio = gen_audio[0].numpy()

        score, *_ = self.metric.score(gt_audio, gen_audio)

        return score

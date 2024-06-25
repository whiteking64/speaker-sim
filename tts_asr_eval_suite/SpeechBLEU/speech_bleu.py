import discrete_speech_metrics as dsm
import torchaudio


class SpeechBLEU:
    def __init__(self, device) -> None:
        self.device = device
        self.sr = 16000
        self.metric = dsm.UTMOS(sr=16000)

    def __call__(self, gen_audio_path):

        gen_audio, gen_sr = torchaudio.load(gen_audio_path)

        if gen_sr != self.sr:
            gen_audio = torchaudio.functional.resample(gen_audio, gen_sr, self.sr)

        gen_audio = gen_audio[0].numpy()

        score = self.metric.score(gen_audio)

        return score

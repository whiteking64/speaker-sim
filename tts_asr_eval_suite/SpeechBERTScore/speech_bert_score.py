import discrete_speech_metrics as dsm
import torchaudio
from pydub import AudioSegment

from tts_asr_eval_suite.utils.utils import torch_rms_norm


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

        ref_dbfs = AudioSegment.from_file(gt_audio_path).dBFS
        gt_audio = torch_rms_norm(gt_audio, db_level=ref_dbfs)
        gen_audio = torch_rms_norm(gen_audio, db_level=ref_dbfs)

        gt_audio = gt_audio[0].numpy()
        gen_audio = gen_audio[0].numpy()

        score, *_ = self.metric.score(gt_audio, gen_audio)

        return score

import os

import nemo.collections.asr as nemo_asr

from tts_asr_eval_suite.cer_wer.cer_wer import CERWER


class ASRIntelligibility:
    def __init__(self, device) -> None:
        self.device = device

        self.asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_en_conformer_transducer_large").to(
            device
        )

        self.cer_wer = CERWER()

    def __call__(self, audio_dir, gt_transcripts_path, output_dir=None):

        if os.path.isdir(audio_dir):
            audio_fnames = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav") or f.endswith(".flac")]
        else:
            raise ValueError(f"Invalid path: {audio_dir}")

        transcripts = self.asr_model.transcribe(audio_fnames, verbose=True, batch_size=32)[0]

        if output_dir is None:
            output_dir = audio_dir
        output_file = os.path.join(output_dir, "asr_transcripts.tsv")

        for audio_fname, transcript in zip(audio_fnames, transcripts):
            with open(output_file, "a") as f:
                f.write(f"{audio_fname}\t{transcript}\n")

        cre_wer_scores = self.cer_wer(output_file, gt_transcripts_path)

        return cre_wer_scores

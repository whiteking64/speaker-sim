import re

from num2words import num2words

from TTS.tts.layers.xtts.tokenizer import expand_numbers_multilingual
from faster_whisper import WhisperModel

from tts_asr_eval_suite.cer_wer.cer_wer import CERWER


def custom_expand_numbers_multilingual(text, lang):
    # if coqui TTS number expands fails, uses num2words
    try:
        text = expand_numbers_multilingual(text, lang)
    except:
        if lang == "cs":
            lang = "cz"

        numbers = re.findall(r'\d+', text)
        # Transliterate the numbers to text
        for num in numbers:
            try:
                transliterated_num = ''.join(num2words(int(num), lang=lang))
            except:
                transliterated_num = num
            text = text.replace(num, transliterated_num, 1)
    return text


class FasterWhisperSTT(object):
    def __init__(self, model_name="large-v3", use_cuda=False) -> None:
        self.model = WhisperModel(model_name, device='cuda' if use_cuda else 'cpu', compute_type="float16")
        self.segments = None

    def transcribe_audio(self, audio, language=None):
        segments, _ = self.model.transcribe(audio, beam_size=5, language=language)
        segments = list(segments)
        self.segments = segments
        transcription = "".join([segment.text for segment in segments])
        # convert number to words
        transcription = custom_expand_numbers_multilingual(transcription, lang=language)
        return transcription

    def get_segments(self):
        return self.segments


class ASRIntelligibility:
    def __init__(self, device) -> None:
        self.device = device
        self.transcriber = FasterWhisperSTT("large-v3", use_cuda=device == "cuda")
        self.cer_wer = CERWER()

    def __call__(self, pred_audio, gt_transcript, language="en"):
        transcription = self.transcriber.transcribe_audio(pred_audio, language=language)
        wer, cer = self.cer_wer.run_single(transcription, gt_transcript)
        result = {"WER": wer, "CER": cer}
        return result

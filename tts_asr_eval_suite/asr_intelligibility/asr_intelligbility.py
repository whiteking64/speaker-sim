import re

import librosa
import torch
from TTS.tts.layers.xtts.tokenizer import expand_numbers_multilingual
from faster_whisper import WhisperModel
from num2words import num2words
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, HubertForCTC, Wav2Vec2Tokenizer

from tts_asr_eval_suite.cer_wer.cer_wer import CERWER

from tts_asr_eval_suite.asr_intelligibility.text_cleaners import clean_text


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
    def __init__(self, device, model_name="large-v3") -> None:
        if isinstance(device, torch.device):
            device_name = device.type
            device_index = device.index
        else:
            device_name = device
            device_index = 0

        # print(f"Device: {device_name}, Device Index: {device_index}", type(device_name), type(device_index))
        self.model = WhisperModel(model_name, device=device_name, device_index=device_index, compute_type="default")
        self.segments = None

    def transcribe_audio(self, audio, language=None):
        segments, _ = self.model.transcribe(audio, beam_size=5, language=language)
        segments = list(segments)
        self.segments = segments
        transcription = "".join([segment.text for segment in segments])
        return transcription

    def get_segments(self):
        return self.segments


class Wav2VecSTT(object):
    def __init__(self, model_name="facebook/wav2vec2-large-960h-lv60-self", device='cpu', sr=16000) -> None:
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
        self.sr = sr

    def cuda(self):
        self.model = self.model.cuda()

    def cpu(self):
        self.model = self.model.cpu()

    def transcribe_audio(self, audio, language=None):
        input_audio, _ = librosa.load(audio, sr=self.sr)
        input_values = self.tokenizer(input_audio, return_tensors="pt", padding="longest").input_values

        input_values = input_values.to(self.model.device)

        with torch.no_grad():
            logits = self.model(input_values).logits
            # arg softmax  t oget the most probablies tokens
            predicted_ids = torch.argmax(logits, dim=-1)

        # decode
        transcription = self.tokenizer.batch_decode(predicted_ids)[0]

        return transcription

    def get_segments(self):
        return None


class HuBERTSTT(object):
    def __init__(self, model_name="facebook/hubert-large-ls960-ft", device='cpu', sr=16000) -> None:
        self.tokenizer = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = HubertForCTC.from_pretrained(model_name).to(device)
        self.sr = sr

    def cuda(self):
        self.model = self.model.cuda()

    def cpu(self):
        self.model = self.model.cpu()

    def transcribe_audio(self, audio, language=None):
        input_audio, _ = librosa.load(audio, sr=self.sr)
        input_values = self.tokenizer(input_audio, return_tensors="pt", padding="longest").input_values

        input_values = input_values.to(self.model.device)

        with torch.no_grad():
            logits = self.model(input_values).logits
            # arg softmax  t oget the most probablies tokens
            predicted_ids = torch.argmax(logits, dim=-1)

        # decode
        transcription = self.tokenizer.batch_decode(predicted_ids)[0]

        return transcription

    def get_segments(self):
        return None


class ASRIntelligibility:
    def __init__(self, device, method) -> None:
        self.device = device
        self.method = method
        if method == "wav2vec":
            self.transcribers = {
                "wav2vec": Wav2VecSTT(device=device)
            }
        elif method == "hubert":
            self.transcribers = {
                "hubert": HuBERTSTT(device=device)
            }
        elif method == "whisper":
            self.transcribers = {
                "whisper": FasterWhisperSTT(device=device)
            }
        elif method == 'all':
            self.transcribers = {
                "wav2vec": Wav2VecSTT(device=device),
                "hubert": HuBERTSTT(device=device),
                "whisper": FasterWhisperSTT(device=device)
            }
        else:
            raise ValueError("Invalid ASR method")
        self.cer_wer = CERWER()

    def __call__(self, pred_audio, gt_transcript=None, reference_audio=None, language="en"):
        assert gt_transcript is not None or reference_audio is not None, \
            "Either ground truth transcript or reference audio must be provided"
        results = {}
        transcriptions = {}
        if gt_transcript is not None:
            gt_transcript = clean_text(gt_transcript)
        for method, transcriber in self.transcribers.items():

            transcription = transcriber.transcribe_audio(pred_audio, language=language)

            transcription = clean_text(transcription)

            if gt_transcript is None:
                gt_transcript = transcriber.transcribe_audio(reference_audio, language=language)
                gt_transcript = clean_text(gt_transcript)

            wer, cer = self.cer_wer.run_single(transcription, gt_transcript)
            results[f"WER ({method})"] = wer
            results[f"CER ({method})"] = cer
            transcriptions[f"Transcription ({method})"] = transcription

        results["WER (avg)"] = sum([results[f"WER ({method})"] for method in self.transcribers]) / len(self.transcribers)
        results["CER (avg)"] = sum([results[f"CER ({method})"] for method in self.transcribers]) / len(self.transcribers)

        return results, transcriptions, gt_transcript

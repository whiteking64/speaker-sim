import re

import librosa
import torch
from TTS.tts.layers.xtts.tokenizer import expand_numbers_multilingual
from faster_whisper import WhisperModel
from num2words import num2words
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor, HubertForCTC

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


class Wav2VecSTT(object):
    def __init__(self, model_name="facebook/wav2vec2-large-960h-lv60-self", use_cuda=False, sr=16000) -> None:
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.sr = sr
        if use_cuda:
            self.model = self.model.cuda()

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

        # convert number to words
        transcription = custom_expand_numbers_multilingual(transcription, lang=language)
        return transcription

    def get_segments(self):
        return None


class HuBERTSTT(object):
    def __init__(self, model_name="facebook/hubert-large-ls960-ft", use_cuda=False, sr=16000) -> None:
        self.tokenizer = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = HubertForCTC.from_pretrained(model_name)
        self.sr = sr
        if use_cuda:
            self.model = self.model.cuda()

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

        # convert number to words
        transcription = custom_expand_numbers_multilingual(transcription, lang=language)
        return transcription

    def get_segments(self):
        return None


class ASRIntelligibility:
    def __init__(self, device, method) -> None:
        self.device = device
        self.method = method
        if method == "wav2vec":
            self.transcribers = {
                "wav2vec": Wav2VecSTT(use_cuda=device == "cuda")
            }
        elif method == "hubert":
            self.transcribers = {
                "hubert": HuBERTSTT(use_cuda=device == "cuda")
            }
        elif method == "whisper":
            self.transcribers = {
                "whisper": FasterWhisperSTT(use_cuda=device == "cuda")
            }
        elif method == 'all':
            self.transcribers = {
                "wav2vec": Wav2VecSTT(use_cuda=device == "cuda"),
                "hubert": HuBERTSTT(use_cuda=device == "cuda"),
                "whisper": FasterWhisperSTT(use_cuda=device == "cuda")
            }
        else:
            raise ValueError("Invalid ASR method")
        self.cer_wer = CERWER()

    def __call__(self, pred_audio, gt_transcript, language="en"):
        results = {}
        transcriptions = {}
        for method, transcriber in self.transcribers.items():
            transcription = transcriber.transcribe_audio(pred_audio, language=language)
            wer, cer = self.cer_wer.run_single(transcription, gt_transcript)
            results[f"WER ({method})"] = wer
            results[f"CER ({method})"] = cer
            transcriptions[f"Transcription ({method})"] = transcription

        results["WER (avg)"] = sum([results[f"WER ({method})"] for method in self.transcribers]) / len(self.transcribers)
        results["CER (avg)"] = sum([results[f"CER ({method})"] for method in self.transcribers]) / len(self.transcribers)

        return results, transcriptions

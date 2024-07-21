import jiwer
import jiwer.transforms as tr
from packaging import version
import importlib.metadata as importlib_metadata

import unicodedata
import sys

ALL_PUNCTUATION = "".join((chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P')))

SENTENCE_DELIMITER = ""

cer_transform = tr.Compose(
    [
        tr.RemoveMultipleSpaces(),
        tr.RemoveWhiteSpace(replace_by_space=True),
        tr.RemovePunctuation(),
        tr.Strip(),
        tr.ReduceToSingleSentence(SENTENCE_DELIMITER),
        tr.ReduceToListOfListOfChars(),
    ]
)

wer_transform = tr.Compose(
    [
        tr.RemoveMultipleSpaces(),
        tr.RemoveWhiteSpace(replace_by_space=True),
        tr.RemovePunctuation(),
        tr.Strip(),
        tr.ReduceToSingleSentence(SENTENCE_DELIMITER),
        tr.ReduceToListOfListOfWords(),
    ]
)


def compute_cer(reference, hypothesis):
    reference = reference.lower()
    hypothesis = hypothesis.lower()
    cer = jiwer.wer(reference, hypothesis, truth_transform=cer_transform, hypothesis_transform=cer_transform)
    return cer


def compute_wer(reference, hypothesis):
    reference = reference.lower()
    hypothesis = hypothesis.lower()
    wer = jiwer.wer(reference, hypothesis, truth_transform=wer_transform, hypothesis_transform=wer_transform)
    return wer


def normalize_text(text):
    # remove ponctuation
    text = text.translate(str.maketrans('', '', ALL_PUNCTUATION))
    text = text.lower()
    text = ' '.join(text.split())
    if not text:
        text = " "
    return text

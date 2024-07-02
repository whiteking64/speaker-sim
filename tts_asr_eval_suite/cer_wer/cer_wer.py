import os

import jiwer

from tts_asr_eval_suite.cer_wer.funcs import compute_wer, compute_cer


def normalize_sentence(sentence):
    """Normalize sentence"""
    # Convert all characters to upper.
    sentence = sentence.upper()
    # Delete punctuations.
    sentence = jiwer.RemovePunctuation()(sentence)
    # Remove \n, \t, \r, \x0c.
    sentence = jiwer.RemoveWhiteSpace(replace_by_space=True)(sentence)
    # Remove multiple spaces.
    sentence = jiwer.RemoveMultipleSpaces()(sentence)
    # Remove white space in two end of string.
    sentence = jiwer.Strip()(sentence)

    # Convert all characters to upper.
    sentence = sentence.upper()

    return sentence


class CERWER:
    def __call__(self, pred_transcripts_path, gt_transcripts_path):
        """

        :param pred_transcripts_path: tsv file with audio file path and transcript
        :param gt_transcripts_path:  tsv file with audio file path and transcript
        :return:
        """

        with open(gt_transcripts_path, "r") as f:
            gt_transcripts = {line.split("\t")[0]: line.split("\t")[1] for line in f.readlines()}

        with open(pred_transcripts_path, "r") as f:
            pred_transcripts = {line.split("\t")[0]: line.split("\t")[1] for line in f.readlines()}

        results = {'wer': {}, 'cer': {}}

        for fname in gt_transcripts.keys():
            wer = compute_wer(gt_transcripts[fname], pred_transcripts[fname])
            cer = compute_cer(gt_transcripts[fname], pred_transcripts[fname])

            results['wer'][fname] = wer
            results['cer'][fname] = cer

        return results

    @staticmethod
    def run_single(pred, gt):
        wer = compute_wer(gt, pred)
        cer = compute_cer(gt, pred)
        return wer, cer

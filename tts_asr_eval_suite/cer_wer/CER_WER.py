import os

import jiwer


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

        output_file = os.path.join(os.path.dirname(pred_transcripts_path), "wer_cer.tsv")

        if os.path.exists(output_file):
            # Remove the file if it already exists
            os.remove(output_file)

        for fname in gt_transcripts.keys():
            wer = jiwer.wer(normalize_sentence(gt_transcripts[fname]), normalize_sentence(pred_transcripts[fname]))
            cer = jiwer.cer(normalize_sentence(gt_transcripts[fname]), normalize_sentence(pred_transcripts[fname]))

            with open(output_file, "a") as f:
                f.write(f"{fname}\t{wer}\t{cer}\n")

        return output_file

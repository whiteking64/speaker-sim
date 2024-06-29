import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import tts_asr_eval_suite as te
from tts_asr_eval_suite.utils.utils import bootstrap_ci_df


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('pred_transcript_file', type=str, help='Input directory containing audio files')
    argparser.add_argument('gt_transcript_file', type=str, help='Ground truth transcript tsv file')
    argparser.add_argument('out_filepath', type=str, help='Output filepath to save the results')
    argparser.add_argument('--rerun', action='store_true', help='Rerun the evaluation even if the results file exists')

    args = argparser.parse_args()

    # Define the metrics to evaluate
    metric = te.CERWER()

    if args.rerun or not Path(args.out_filepath).exists():
        results = metric(pred_transcripts_path=args.pred_transcript_file, gt_transcripts_path=args.gt_transcript_file)
        df = pd.DataFrame(results)
        df.to_csv(args.out_filepath, index=False)

    # Load the results
    df = pd.read_csv(args.out_filepath)

    # get confidence intervals and print the results
    cer_result = bootstrap_ci_df(df, 'cer', metric_func=np.mean, B=5000, alpha=0.05)
    wer_result = bootstrap_ci_df(df, 'wer', metric_func=np.mean, B=5000, alpha=0.05)

    print(f"ASR (CER): {cer_result}")
    print(f"ASR (WER): {wer_result}")


if __name__ == '__main__':

    main()

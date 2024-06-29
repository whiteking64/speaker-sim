import argparse
import concurrent.futures
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import tts_asr_eval_suite as te
from tts_asr_eval_suite.utils.utils import bootstrap_ci_df


def process_file(metrics, pred_dir, ref_dir, filename):
    result = {}
    pred_path = os.path.join(pred_dir, filename)
    ref_path = os.path.join(ref_dir, filename)

    assert os.path.exists(pred_path) and os.path.exists(ref_path), f"File {filename} does not exist in both directories"
    try:
        for name, model in metrics.items():
            result[name] = model(gt_audio_path=ref_path, pred_audio_path=pred_path)

        return pred_path, result
    except Exception as exc:
        print(f'Files ({pred_path}, {ref_path}) generated an exception: {exc}')
        return pred_path, None


def process_directory(pred_dir, ref_dir, metrics):
    results = {}
    file_names = [filename for filename in os.listdir(pred_dir)
                  if filename.endswith('.wav') or filename.endswith('.flac')]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Use dict comprehension to map futures to filepaths
        future_to_file = {executor.submit(process_file, metrics, pred_dir, ref_dir, filename): filename for filename in file_names}

        # Process as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(file_names)):
            filepath = future_to_file[future]
            try:
                _, scores = future.result()
                if scores is not None:
                    fname = os.path.basename(filepath).replace('.wav', "")
                    for name, score in scores.items():
                        if name not in results:
                            results[name] = {}
                        results[name][fname] = score
            except Exception as exc:  # This is just a precaution, exceptions should be caught in process_file
                print(f'File {filepath} generated an exception: {exc}')

    return results


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('pred_dir', type=str, help='Input directory containing audio files')
    argparser.add_argument('ref_dir', type=str, help='Reference directory containing reference audio files')
    argparser.add_argument('out_filepath', type=str, help='Output filepath to save the results')
    argparser.add_argument('--rerun', action='store_true', help='Rerun the evaluation even if the results file exists')

    args = argparser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define the metrics to evaluate
    metrics = {
        'LogF0RMSE': te.LogF0RMSE(device),
        'MelCepstralDistortion': te.MelCepstralDistortion(device),
        'SpeechBERTScore': te.SpeechBERTScore(device),
        'SpeechBLEU': te.SpeechBLEU(device),
        'SpeechTokenDistance': te.SpeechTokenDistance(device),
    }

    if args.rerun or not Path(args.out_filepath).exists():
        results = process_directory(args.inp_dir, args.ref_dir, metrics)
        df = pd.DataFrame(results)
        df.to_csv(args.out_filepath, index=False)

    # Load the results
    df = pd.read_csv(args.out_filepath)

    # get confidence intervals and print the results
    for metric in metrics.keys():
        result = bootstrap_ci_df(df, metric, metric_func=np.mean, B=5000, alpha=0.05)
        print(f"{metric}: {result}")


if __name__ == '__main__':

    main()

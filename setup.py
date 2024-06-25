from setuptools import setup, find_packages

setup(
    name="tts_asr_eval_suite",
    version="0.1",
    packages=find_packages(),
    package_dir={"": "."},
    install_requires=[
        "torch",
        "fairseq",
        "lightning",
        "packaging",
        "omegaconf",
        "s3prl",
        "onnxruntime",
        "librosa",
        "soundfile",
        "requests",
        "tqdm",
        "gdown",
        "nemo_toolkit[asr]",
        "jiwer",
        "discrete-speech-metrics @ git+https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics.git",
    ],
)

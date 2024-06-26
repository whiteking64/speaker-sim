from setuptools import setup, find_packages

try:
    from pypesq import pesq
except ImportError:
    print("Please first install pypesq (pip install pypesq@https://github.com/vBaiCai/python-pesq/archive/master.zip)")

setup(
    name="tts_asr_eval_suite",
    version="0.1",
    packages=find_packages(),
    package_dir={"": "."},
    include_package_data=True,
    package_data={
        'tts_asr_eval_suite.dnsmos': ['ckpt/DNSMOS/*.onnx', 'ckpt/pDNSMOS/*.onnx'],
        'tts_asr_eval_suite.secs': ['ckpt/*'],
    },
    install_requires=[
        "cython",
        "protobuf<5",
        "torch",
        "lightning",
        "packaging",
        "omegaconf",
        "s3prl @ git+https://github.com/s3prl/s3prl.git",
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

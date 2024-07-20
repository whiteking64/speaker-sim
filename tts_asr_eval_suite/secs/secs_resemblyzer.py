from resemblyzer import preprocess_wav, VoiceEncoder


class SECSResemblyzer:
    def __init__(self, device) -> None:
        self.device = device
        self.sv_model = VoiceEncoder(device=device, verbose=False)
        self.sr = 16000

    def __call__(self, prompt_audio, gen_audio):

        prompt_audio = preprocess_wav(prompt_audio.squeeze().cpu().numpy(), source_sr=self.sr)
        gen_audio = preprocess_wav(gen_audio.squeeze().cpu().numpy(), source_sr=self.sr)

        prompt_embedding = self.sv_model.embed_utterance(prompt_audio)
        gen_embedding = self.sv_model.embed_utterance(gen_audio)

        similarity = prompt_embedding @ gen_embedding

        return similarity

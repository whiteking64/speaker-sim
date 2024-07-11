from resemblyzer import preprocess_wav, VoiceEncoder


class SECSResemblyzer:
    def __init__(self, device) -> None:
        self.device = device
        self.sv_model = VoiceEncoder(device=device, verbose=False)

    def __call__(self, prompt_path, gen_path):

        prompt_audio = preprocess_wav(prompt_path)
        gen_audio = preprocess_wav(gen_path)

        prompt_embedding = self.sv_model.embed_utterance(prompt_audio)
        gen_embedding = self.sv_model.embed_utterance(gen_audio)

        similarity = prompt_embedding @ gen_embedding

        return similarity

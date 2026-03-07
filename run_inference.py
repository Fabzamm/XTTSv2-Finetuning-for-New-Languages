import torch
import torchaudio
from tqdm import tqdm
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from IPython.display import Audio
import argparse

def create_parser():
    parser = argparse.ArgumentParser(description="XTTS Inference")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config.json")
    parser.add_argument("--vocab", type=str, required=True,
                        help="Path to vocab.json")
    parser.add_argument("--speaker_audio", type=str, required=True,
                        help="Path to speaker reference audio (.wav)")
    parser.add_argument("--text", type=str, required=True,
                        help="Text to synthesize")
    parser.add_argument("--language", type=str, default="mt",
                        help="Language code")
    parser.add_argument("--output_path", type=str, default="output.wav",
                        help="Path to save output audio")
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=5.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.85)

    return parser

def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = XttsConfig()
    config.load_json(args.config)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=args.checkpoint, vocab_path=args.vocab, use_deepspeed=False)
    model.to(device)
    print("✓ Model loaded!")

    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=args.speaker_audio,
        gpt_cond_len=model.config.gpt_cond_len,
        max_ref_length=model.config.max_ref_len,
        sound_norm_refs=model.config.sound_norm_refs,
    )

    wav_chunks = []
    for text in tqdm([args.text]):
        # Show what the model receives
        tokens = model.tokenizer.encode(text, lang=args.language)
        decoded = model.tokenizer.decode(tokens)
        print(f"Original text : {text}")
        print(f"Token IDs     : {tokens}")
        print(f"Decoded back  : {decoded}")
        print(f"Token count   : {len(tokens)}")
        wav_chunk = model.inference(
            text=text, language=args.language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=args.temperature,
            length_penalty=args.length_penalty,
            repetition_penalty=args.repetition_penalty,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        wav_chunks.append(torch.tensor(wav_chunk["wav"]))

    out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0).cpu()
    torchaudio.save(args.output_path, out_wav, 24000)
    print(f"✓ Audio saved to {args.output_path}")

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    run_inference(args)

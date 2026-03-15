import argparse
from tokenizers import Tokenizer
import os
import pandas as pd
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
import json
from TTS.tts.layers.xtts.tokenizer import multilingual_cleaners
from masri.tokeniser.km_tokeniser import KMTokeniser
from TTS.tts.utils.text.cleaners import collapse_whitespace

def combine_tokenizers(old_tokenizer, new_tokenizer, save_dir):
    # Load both the json files, take the union, and store it
    json1 = json.load(open(os.path.join(old_tokenizer, 'vocab.json')))
    json2 = json.load(open(os.path.join(new_tokenizer, 'vocab.json')))

    # Create a new vocabulary
    new_vocab = {}
    idx = 0
    for word in json1.keys():
        if word not in new_vocab.keys():
            new_vocab[word] = idx
            idx += 1

    # Add words from second tokenizer
    for word in json2.keys():
        if word not in new_vocab.keys():
            new_vocab[word] = idx
            idx += 1

    # Make the directory if necessary
    os.makedirs(save_dir, exist_ok=True)

    # Save the vocab
    with open(os.path.join(save_dir, 'vocab.json'), 'w') as fp:
        json.dump(new_vocab, fp, ensure_ascii=False)

    # Merge the two merges file. Don't handle duplicates here
    # Concatenate them, but ignore the first line of the second file
    os.system('cat {} > {}'.format(os.path.join(old_tokenizer, 'merges.txt'), os.path.join(save_dir, 'merges.txt')))
    os.system('tail -n +2 -q {} >> {}'.format(os.path.join(new_tokenizer, 'merges.txt'), os.path.join(save_dir, 'merges.txt')))


def extend_tokenizer(args):
    
    root = os.path.join(args.output_path, "XTTS_v2.0_original_model_files/")

    # save seperately vocab, merges
    existing_tokenizer = Tokenizer.from_file(os.path.join(root, "vocab.json"))
    old_tokenizer_path = os.path.join(root, "old_tokenizer/")
    os.makedirs(old_tokenizer_path, exist_ok=True)
    existing_tokenizer.model.save(old_tokenizer_path)

    # train new tokenizer
    tokeniser = KMTokeniser()

    def clean_text(text):
        text = multilingual_cleaners(text, "mt")
        tokens = tokeniser.tokenise(text)
        text = " ".join(tokens)
        text = collapse_whitespace(text)
        return text
    
    def text_iterator():
        # Source 1: always use the CSV
        df = pd.read_csv(args.metadata_path, sep="|")
        for text in df["text"].dropna().tolist():
            yield clean_text(text)
    
        # Source 2: only if --use_korpus is passed
        if args.use_korpus:
            from datasets import load_dataset
            korpus = load_dataset(
                "MLRS/korpus_malti",
                "shuffled_sampled",
                revision="4.3.alpha",
                split="train",
                streaming=True
            )
            for i, row in enumerate(korpus):
                if i >= args.korpus_max_samples:
                    break
                yield clean_text(row["text"])

    new_tokenizer = Tokenizer(BPE())
    new_tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(special_tokens=[f"[{args.language}]"], vocab_size=args.extended_vocab_size)
    new_tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    new_tokenizer.add_special_tokens([f"[{args.language}]"])

    new_tokenizer_path = os.path.join(root, "new_tokenizer/")
    os.makedirs(new_tokenizer_path, exist_ok=True)
    new_tokenizer.model.save(new_tokenizer_path)

    merged_tokenizer_path = os.path.join(root, "merged_tokenizer/")
    combine_tokenizers(
        old_tokenizer_path,
        new_tokenizer_path,
        merged_tokenizer_path
    )

    tokenizer = Tokenizer.from_file(os.path.join(root, "vocab.json"))
    tokenizer.model = tokenizer.model.from_file(os.path.join(merged_tokenizer_path, 'vocab.json'), os.path.join(merged_tokenizer_path, 'merges.txt'))
    tokenizer.add_special_tokens([f"[{args.language}]"])

    tokenizer.save(os.path.join(root, "vocab.json"))

    os.system(f'rm -rf {old_tokenizer_path} {new_tokenizer_path} {merged_tokenizer_path}')

def adjust_config(args):
    config_path = os.path.join(args.output_path, "XTTS_v2.0_original_model_files/config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    config["languages"] += [args.language]
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_path", type=str, required=True, help="")
    parser.add_argument("--metadata_path", type=str, required=True, help="")
    parser.add_argument("--language", type=str, required=True, help="")
    parser.add_argument("--extended_vocab_size", default=2000, type=int, required=True, help="")
    parser.add_argument("--use_korpus", action="store_true", help="Also train on korpus_malti")
    parser.add_argument("--korpus_max_samples", type=int, default=500_000, help="Max sentences from korpus_malti")

    args = parser.parse_args()

    extend_tokenizer(args)
    adjust_config(args)

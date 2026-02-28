import sys
sys.path.insert(0, r'C:\Users\HP\FYP\Github code\XTTSv2-Finetuning-for-New-Languages')

from TTS.tts.layers.xtts.tokenizer import multilingual_cleaners
from masri.transcribe.num2text import num2text as mt_num2text

cases = [
    (1,       "wieħed"),
    (2,       "tnejn"),
    (10,      "għaxra"),
    (11,      "ħdax"),
    (20,      "għoxrin"),
    (21,      "wieħed u għoxrin"),
    (100,     "mija"),
    (301,     "tliet mija u wieħed"),
    (1000,    "elf"),
    (35000,  "ħamsa u tletin elf"),
    (1000000, "miljun"),
]

print("=== Testing mt_num2text ===")
for num, expected in cases:
    result = mt_num2text(num).strip()
    status = "✓" if result == expected else "✗"
    print(f"{status} {num} → '{result}'" + (f" (expected '{expected}')" if result != expected else ""))
    
    
print("=== Testing thousands separator handling ===\n")

comma_cases = [
    ("€35,000 huwa ħafna flus.",  "euro with thousands separator"),
    ("€1,000,000 fil-bank.",      "dollar with millions"),
    ("Kien hemm 35,000 ruħ.",     "plain number with thousands separator"),
    ("1,000,000,000 huwa numru kbir.", "plain number with multiple commas"),
]

for text_in, desc in comma_cases:
    result = multilingual_cleaners(text_in, "mt")
    print(f"  [{desc}]")
    print(f"    IN  → {text_in}")
    print(f"    OUT → {result}\n")

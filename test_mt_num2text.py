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
    (1000000, "miljun"),
]

print("=== Testing mt_num2text ===")
for num, expected in cases:
    result = mt_num2text(num).strip()
    status = "✓" if result == expected else "✗"
    print(f"{status} {num} → '{result}'" + (f" (expected '{expected}')" if result != expected else ""))
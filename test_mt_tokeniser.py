from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer, multilingual_cleaners

# ── Test multilingual_cleaners directly ──────────────────────────────────────

def test_cleaners():
    cases = [
        # (input, expected_output, description)
        ("Il-kelb huwa kbir.", "il-kelb huwa kbir.", "lowercase + definite article"),
        ("€20 jekk jogħġbok.", "għoxrin ewro jekk jogħġbok.", "euro symbol swap"),
        ("$50 huma wisq.", "ħamsin dollaru huma wisq.", "dollar symbol swap"),
        ("Hemm   spazju   żejjed.", "hemm spazju żejjed.", "collapse whitespace"),
        ("Il-prezz huwa €20.50.", "il-prezz huwa għoxrin ewro u ħamsin ċenteżmu.", "euro swap with decimals"),
        ("Għandi 3 klieb.", "għandi tlieta klieb.", "number to Maltese words"),
        ("Kien hemm 50 suldat.", "kien hemm ħamsin suldat.", "larger number"),
        ("Dr. Smith huwa hawn.", "doktor smith huwa hawn.", "abbreviation expansion"),
        ("100% korrett.", "mija fil-mija korrett.", "percent symbol"),
        ("Żur il-paġna @ dan is-sit.", "żur il-paġna at dan is-sit.", "at symbol"),
    ]

    print("=== Testing multilingual_cleaners ===")
    for text_in, expected, desc in cases:
        result = multilingual_cleaners(text_in, "mt")
        status = "✓" if result == expected else "✗"
        print(f"{status} {desc}")
        if result != expected:
            print(f"    Input:    {text_in}")
            print(f"    Expected: {expected}")
            print(f"    Got:      {result}")


# ── Test preprocess_text (includes KMTokeniser) ───────────────────────────────

def test_preprocess():
    from TTS.tts.layers.xtts.tokenizer import DEFAULT_VOCAB_FILE
    tokenizer = VoiceBpeTokenizer(vocab_file=DEFAULT_VOCAB_FILE)

    cases = [
        ("L-għalliem huwa hawn.", "l- għalliem huwa hawn.", "definite article split"),
        ("Sant'Anna hija sabiħa.", "sant' anna hija sabiħa.", "proclitic sant'"),
        ("Il-10/05/2024 kien jum sabiħ.", "il- 10/05/2024 kien jum sabiħ.", "date preservation"),
        ("Ħdax-il tifel.", "ħdax -il tifel.", "definite numeral"),
    ]

    print("\n=== Testing preprocess_text (with KMTokeniser) ===")
    for text_in, expected, desc in cases:
        result = tokenizer.preprocess_text(text_in, "mt")
        status = "✓" if result == expected else "✗"
        print(f"{status} {desc}")
        if result != expected:
            print(f"    Input:    {text_in}")
            print(f"    Expected: {expected}")
            print(f"    Got:      {result}")


# ── Test full encode/decode round-trip ────────────────────────────────────────

def test_encode_decode():
    from TTS.tts.layers.xtts.tokenizer import DEFAULT_VOCAB_FILE
    tokenizer = VoiceBpeTokenizer(vocab_file=DEFAULT_VOCAB_FILE)

    sentences = [
        "Il-kelb huwa kbir.",
        "Għandi €20 fil-but.",
        "Il-10 ta' Mejju kien jum sabiħ.",
        "Dr. Borg qal li l-pazjent huwa tajjeb.",
    ]

    print("\n=== Testing encode/decode round-trip ===")
    for sentence in sentences:
        try:
            ids = tokenizer.encode(sentence, "mt")
            decoded = tokenizer.decode(ids)
            print(f"✓ '{sentence}'")
            print(f"    Tokens: {ids[:10]}{'...' if len(ids) > 10 else ''}")
            print(f"    Decoded: '{decoded}'")
        except Exception as e:
            print(f"✗ '{sentence}'")
            print(f"    Error: {e}")


if __name__ == "__main__":
    test_cleaners()
    test_preprocess()
    test_encode_decode()
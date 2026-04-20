# utils/language_config.py
# --------------------------------------------------------
# Defines supported languages and their character sets
# --------------------------------------------------------

LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
}

# English character list
CHARSET_EN = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,!?;:'\"()-/"
)

# Hindi character list kept compatible with existing checkpoints
CHARSET_HI = (
    "अआइईउऊऋएऐओऔ"
    "कखगघङचछजझञ"
    "टठडढणतथदधन"
    "पफबभमयरलव"
    "शषसह"
    "ािीुूृेैोौंःँ्"
    "0123456789"
)

# Tamil - U+0B80-U+0BFF
CHARSET_TA = "".join(chr(codepoint) for codepoint in range(0x0B80, 0x0C00))

# Telugu - U+0C00-U+0C7F
CHARSET_TE = "".join(chr(codepoint) for codepoint in range(0x0C00, 0x0C80))


def get_charset(lang_code: str) -> str:
    """Return the character set for a given language."""
    if lang_code == "en":
        return CHARSET_EN
    if lang_code == "hi":
        return CHARSET_HI
    if lang_code == "ta":
        return CHARSET_TA
    if lang_code == "te":
        return CHARSET_TE
    raise ValueError(f"Unsupported language code: {lang_code}")





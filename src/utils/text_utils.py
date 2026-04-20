# utils/text_utils.py
# --------------------------------------------------------
# Text <-> Index conversion for OCR models
# --------------------------------------------------------

# THIS LINE IS NOW FIXED to use an absolute import
from src.utils.language_config import get_charset

class TextEncoder:
    def __init__(self, lang_code: str):
        self.charset = get_charset(lang_code)
        self.blank_index = 0
        # Assign 1..N for characters; 0 reserved for CTC blank
        self.char2idx = {ch: i+1 for i, ch in enumerate(self.charset)}
        self.idx2char = {i+1: ch for i, ch in enumerate(self.charset)}

    def encode(self, text: str):
        """Convert text to list of indices."""
        # Ensure that characters not in the charset are ignored
        return [self.char2idx[ch] for ch in text if ch in self.char2idx]

    def decode(self, indices):
        """CTC-style greedy decoding: collapse repeats, remove blanks."""
        result = []
        prev = None
        for i in indices:
            if i != self.blank_index and i != prev:
                # Use .get() to handle potential invalid indices gracefully
                result.append(self.idx2char.get(i, ""))
            prev = i
        return "".join(result)

    def vocab_size(self):
        return len(self.charset) + 1  # +1 for CTC blank
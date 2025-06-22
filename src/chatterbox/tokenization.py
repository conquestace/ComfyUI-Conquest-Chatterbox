class PromptTokenizer:
    """Simple tokenizer that splits text into fixed-size character tokens."""

    def __init__(self, token_length: int = 300):
        self.token_length = token_length

    def tokenize(self, text: str):
        return [text[i:i + self.token_length] for i in range(0, len(text), self.token_length)]

    __call__ = tokenize

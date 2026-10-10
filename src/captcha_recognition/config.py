from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HybridConfig:
    alphabet: str = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    height: int = 48
    width: int = 320
    decoder_width: int = 256
    decoder_layers: int = 2
    decoder_heads: int = 8
    dropout: float = .1
    max_length: int = 16

    def __post_init__(self):
        if not self.alphabet or len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError('Alphabet must contain unique characters')
        if min(self.height, self.width) < 16 or self.height % 16 or self.width % 16:
            raise ValueError('Input geometry must use positive multiples of 16')
        if min(self.decoder_width, self.decoder_layers, self.decoder_heads, self.max_length) < 1 or not 0 <= self.dropout < 1:
            raise ValueError('Invalid decoder settings')
        if self.decoder_width % 4 or self.decoder_width % self.decoder_heads:
            raise ValueError('Decoder width must be divisible by four and by attention heads')

    def to_dict(self):
        return asdict(self)

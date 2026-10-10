import torch


class Vocabulary:
    """CTC blank and AR EOS share an output index, but have different semantics."""
    def __init__(self, alphabet):
        if not alphabet or len(set(alphabet)) != len(alphabet):
            raise ValueError('Invalid alphabet')
        self.alphabet = alphabet
        self.ids = {char: i for i, char in enumerate(alphabet)}
        self.blank = self.eos = len(alphabet)
        self.bos, self.pad = len(alphabet) + 1, len(alphabet) + 2

    def encode(self, labels, device=None):
        rows = [[self.ids[c] for c in label] for label in labels]
        length = max(map(len, rows), default=0)
        tokens = torch.full((len(rows), length + 2), self.pad, dtype=torch.long, device=device)
        tokens[:, 0] = self.bos
        for index, row in enumerate(rows):
            tokens[index, 1:len(row)+1] = torch.tensor(row, dtype=torch.long, device=device)
            tokens[index, len(row)+1] = self.eos
        return tokens

    def ctc_decode(self, logits):
        labels = []
        for row in logits.argmax(-1).tolist():
            output, previous = [], None
            for token in row:
                if token != previous and token != self.blank:
                    output.append(self.alphabet[token])
                previous = token
            labels.append(''.join(output))
        return labels

    def ar_decode(self, ids):
        labels = []
        for row in ids.tolist():
            output = []
            for token in row:
                if token == self.eos:
                    break
                if token < len(self.alphabet):
                    output.append(self.alphabet[token])
            labels.append(''.join(output))
        return labels

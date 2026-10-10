"""Controlled architecture experiment: preserve horizontal detail in the stem.

Same parameter shapes, 62-character vocabulary and CTC head as DFCRv2.
The first convolution strides (2, 1), rather than (2, 2). Width reduction
is 8 rather than 16, giving 24 CTC steps at 64x192. Existing v2 weights
load strictly, but the changed feature distribution requires fine-tuning.
"""
from dense_ctc import DFCRv2


class DFCRv3(DFCRv2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1.stride = (2, 1)

"""Restore explicit recognition architectures without silent fallbacks."""
from pathlib import Path
from dense_ctc import CHAR_SET, DFCRv2
from dense_ctc_wide import DFCRv3
from dense_ctc_attention import DFCRv4


def model_from_checkpoint(checkpoint, upstream=None):
    architecture = checkpoint.get('architecture', 'v2')
    if architecture == 'dfcr_experimental_v1':
        from captcha_recognition.experimental import load_experimental
        return load_experimental(checkpoint, upstream)
    if architecture == 'dfcr_hybrid_v1':
        from captcha_recognition.model import load_hybrid
        return load_hybrid(checkpoint, upstream)
    if architecture not in ('v2', 'v3', 'v4', 'svtrv2_captcha', 'svtrv2_typed_captcha', 'svtrv2_detail_captcha', 'svtrv2_foreground_captcha', 'svtrv2_foreground_fine_captcha'):
        raise ValueError(f'Unsupported architecture: {architecture}')
    if checkpoint.get('alphabet', CHAR_SET) != CHAR_SET:
        raise ValueError('Unsupported character vocabulary')
    if architecture.startswith('svtrv2_'):
        from svtr_adapter import SVTRCaptcha, SVTRTypedCaptcha, SVTRDetailCaptcha, SVTRForegroundCaptcha, SVTRForegroundFineCaptcha, UPSTREAM_COMMIT
        if checkpoint.get('upstream_commit') != UPSTREAM_COMMIT:
            raise ValueError('Unsupported OpenOCR source revision')
        if not isinstance(checkpoint.get('model_config'), dict):
            raise ValueError('SVTR checkpoints require model_config')
        upstream = upstream or Path(__file__).resolve().parents[1] / 'assets/upstream/OpenOCR'
        model_class = {'svtrv2_captcha':SVTRCaptcha,'svtrv2_typed_captcha':SVTRTypedCaptcha,
                       'svtrv2_detail_captcha':SVTRDetailCaptcha,
                       'svtrv2_foreground_captcha':SVTRForegroundCaptcha,
                       'svtrv2_foreground_fine_captcha':SVTRForegroundFineCaptcha}[architecture]
        model = model_class(checkpoint['model_config'], upstream)
    elif architecture == 'v4':
        if not isinstance(checkpoint.get('model_config'), dict):
            raise ValueError('v4 checkpoints require model_config')
        model = DFCRv4(**checkpoint['model_config'])
    else:
        model = (DFCRv2 if architecture == 'v2' else DFCRv3)()
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=True)
    return model

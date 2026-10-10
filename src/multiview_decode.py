"""Source-agnostic RGB/channel views with CTC probability-mixture decoding."""
import torch
from dense_ctc import CHAR_SET


def greedy_ids(sequence,blank):
    result=[]
    last=None
    for value in sequence:
        if value!=last and value!=blank:
            result.append(value)
        last=value
    return tuple(result)


def mixture_decode(logits,alphabet=CHAR_SET):
    """Choose among view-greedy candidates by equal-weight CTC probabilities.

    Scores marginalize all CTC paths for each candidate, not just its greedy
    path. No source identifier, expected length or answer-dependent hint is used.
    Empty strings are treated as abstentions unless every view is empty.
    """
    views,steps,classes=logits.shape
    blank=len(alphabet)
    if classes!=blank+1:
        raise ValueError('Vocabulary mismatch')
    decoded=[greedy_ids(row,blank) for row in logits.argmax(-1).tolist()]
    candidates=sorted(set(row for row in decoded if row)) or [()]
    from captcha_recognition.decoding import ctc_mixture_scores
    scores=ctc_mixture_scores(logits,candidates,blank)
    winner=int(scores.argmax())
    return dict(prediction=''.join(alphabet[i] for i in candidates[winner]),
        view_predictions=[''.join(alphabet[i] for i in row) for row in decoded],
        candidates=[''.join(alphabet[i] for i in row) for row in candidates],
        candidate_log_probabilities=scores.detach().cpu().tolist(),ctc_steps=steps)


def predict_multiview(model,images):
    """Run one original view plus B/G/R views only for truly colored inputs."""
    logits=model(images)
    blank=len(CHAR_SET)
    original=[greedy_ids(row,blank) for row in logits.argmax(-1).tolist()]
    results=[dict(prediction=''.join(CHAR_SET[i] for i in row),ctc_steps=logits.shape[1],views_used=1) for row in original]
    colorful=((images[:,0]-images[:,1]).abs().flatten(1).amax(1)>1e-6)|((images[:,0]-images[:,2]).abs().flatten(1).amax(1)>1e-6)
    indices=colorful.nonzero(as_tuple=False).flatten()
    if indices.numel():
        selected=images[indices]
        extra=torch.stack([selected[:,c:c+1].expand(-1,3,-1,-1) for c in range(3)],dim=1)
        other=model(extra.flatten(0,1)).reshape(len(indices),3,logits.shape[1],logits.shape[2])
        for index,views in zip(indices.tolist(),torch.cat([logits[indices,None],other],dim=1)):
            result=mixture_decode(views)
            result['views_used']=4
            results[index]=result
    return results

import torch
import math

def ctc_beam_search_decoder(log_probs, text_encoder, beam_width=10):
    """
    log_probs: (T, C) torch tensor (already log-softmaxed)
    returns: decoded string
    """

    T, C = log_probs.size()
    blank = 0

    beams = {(): 0.0}  # sequence -> log probability

    for t in range(T):
        new_beams = {}

        for seq, score in beams.items():
            for c in range(C):
                p = log_probs[t, c].item()
                new_score = score + p

                if c == blank:
                    new_seq = seq
                else:
                    if len(seq) > 0 and seq[-1] == c:
                        new_seq = seq
                    else:
                        new_seq = seq + (c,)

                if new_seq in new_beams:
                    new_beams[new_seq] = max(new_beams[new_seq], new_score)
                else:
                    new_beams[new_seq] = new_score

        # Keep top beams
        beams = dict(
            sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width]
        )

    best_seq = max(beams, key=beams.get)
    return text_encoder.decode(list(best_seq))

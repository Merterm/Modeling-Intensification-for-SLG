# coding: utf-8
"""
This module holds various MT evaluation metrics.
"""
from signjoey.external_metrics import sacrebleu
from signjoey.external_metrics import mscoco_rouge
import numpy as np

WER_COST_DEL = 3
WER_COST_INS = 3
WER_COST_SUB = 4


def chrf(references, hypotheses):
    """
    Character F-score from sacrebleu

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    return (
        sacrebleu.corpus_chrf(hypotheses=hypotheses,
                              references=references).score * 100
    )


def bleu(references, hypotheses):
    """
    Raw corpus BLEU from sacrebleu (without tokenization)

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    bleu_scores = sacrebleu.raw_corpus_bleu(
        sys_stream=hypotheses, ref_streams=[references]
    ).scores
    scores = {}
    for n in range(len(bleu_scores)):
        scores["bleu" + str(n + 1)] = bleu_scores[n]
    return scores


def token_accuracy(references, hypotheses, level="word"):
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :param level: segmentation level, either "word", "bpe", or "char"
    :return:
    """
    correct_tokens = 0
    all_tokens = 0
    split_char = " " if level in ["word", "bpe"] else ""
    assert len(hypotheses) == len(references)
    for hyp, ref in zip(hypotheses, references):
        all_tokens += len(hyp)
        for h_i, r_i in zip(hyp.split(split_char), ref.split(split_char)):
            # min(len(h), len(r)) tokens considered
            if h_i == r_i:
                correct_tokens += 1
    return (correct_tokens / all_tokens) * 100 if all_tokens > 0 else 0.0


def sequence_accuracy(references, hypotheses):
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    assert len(hypotheses) == len(references)
    correct_sequences = sum(
        [1 for (hyp, ref) in zip(hypotheses, references) if hyp == ref]
    )
    return (correct_sequences / len(hypotheses)) * 100 if hypotheses else 0.0


def rouge(references, hypotheses):
    rouge_score = 0
    n_seq = len(hypotheses)

    for h, r in zip(hypotheses, references):
        rouge_score += mscoco_rouge.calc_score(
            hypotheses=[h], references=[r]) / n_seq

    return rouge_score * 100


def wer_list(references, hypotheses):
    total_error = total_del = total_ins = total_sub = total_ref_len = 0

    for r, h in zip(references, hypotheses):
        res = wer_single(r=r, h=h)
        total_error += res["num_err"]
        total_del += res["num_del"]
        total_ins += res["num_ins"]
        total_sub += res["num_sub"]
        total_ref_len += res["num_ref"]

    wer = (total_error / total_ref_len) * 100
    del_rate = (total_del / total_ref_len) * 100
    ins_rate = (total_ins / total_ref_len) * 100
    sub_rate = (total_sub / total_ref_len) * 100

    return {
        "wer": wer,
        "del_rate": del_rate,
        "ins_rate": ins_rate,
        "sub_rate": sub_rate,
    }


def wer_single(r, h):
    r = r.strip().split()
    h = h.strip().split()
    edit_distance_matrix = edit_distance(r=r, h=h)
    alignment, alignment_out = get_alignment(r=r, h=h, d=edit_distance_matrix)

    num_cor = np.sum([s == "C" for s in alignment])
    num_del = np.sum([s == "D" for s in alignment])
    num_ins = np.sum([s == "I" for s in alignment])
    num_sub = np.sum([s == "S" for s in alignment])
    num_err = num_del + num_ins + num_sub
    num_ref = len(r)

    return {
        "alignment": alignment,
        "alignment_out": alignment_out,
        "num_cor": num_cor,
        "num_del": num_del,
        "num_ins": num_ins,
        "num_sub": num_sub,
        "num_err": num_err,
        "num_ref": num_ref,
    }


def edit_distance(r, h):
    """
    Original Code from https://github.com/zszyellow/WER-in-python/blob/master/wer.py
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.
    Main algorithm used is dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    """
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8).reshape(
        (len(r) + 1, len(h) + 1)
    )
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                # d[0][j] = j
                d[0][j] = j * WER_COST_INS
            elif j == 0:
                d[i][0] = i * WER_COST_DEL
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitute = d[i - 1][j - 1] + WER_COST_SUB
                insert = d[i][j - 1] + WER_COST_INS
                delete = d[i - 1][j] + WER_COST_DEL
                d[i][j] = min(substitute, insert, delete)
    return d


def bootstrap_bertscore(baseline_hypo, new_hypo, num_samples=3000, sample_size=200):
    import numpy as np
    num_sents = len(baseline_hypo)

    indices = np.random.randint(
        low=0, high=num_sents, size=(num_samples, sample_size))
    out = {}
    baseline_hypo = baseline_hypo.data.numpy()
    new_hypo = new_hypo.data.numpy()
    delta_origin = np.mean(new_hypo) - np.mean(baseline_hypo)
    c = 0

    for index in indices:
        diff = (new_hypo[index]).mean() - (baseline_hypo[index]).mean()

        if diff > delta_origin:
            c += 1
    print("------DOWN BertScore COMPUTATION------")
    print("{} New better confidence : {:.2f}".format(
        "BERTScore", (c/num_samples)))
    out = {"new_pvalue": 1 - (c/num_samples)}
    print()
    return out


def bootstrap_rogue(references, baseline_hypo, new_hypo, num_samples=3000, sample_size=200):
    """
    Original Code adapted from https://github.com/pytorch/translate/tree/master/pytorch_translate
    Attributes:
        references -> the reference data
        baseline_hypo -> the baseline model output
        new_hypo -> new mdoel_hypo
    """

    import numpy as np
    num_sents = len(baseline_hypo)
    assert len(baseline_hypo) == len(new_hypo) == len(references)

    indices = np.random.randint(
        low=0, high=num_sents, size=(num_samples, sample_size))
    out = {}

    baseline_better = 0
    new_better = 0
    num_equal = 0
    for index in indices:
        sub_base_hypo = [y for i, y in enumerate(
            baseline_hypo) if i in index]
        sub_new_hypo = [y for i, y in enumerate(
            new_hypo) if i in index]
        sub_ref = [y for i, y in enumerate(
            references) if i in index]
        baseline_rogue = rouge(sub_ref, sub_base_hypo)
        new_rogue = rouge(sub_ref, sub_new_hypo)

        if new_rogue > baseline_rogue:
            new_better += 1
        elif baseline_rogue > new_rogue:
            baseline_better += 1
        else:
            num_equal += 1
    print("------DOWN ROGUE COMPUTATION------")
    print("{} New better confidence : {:.2f}".format(
        "ROUGUE", new_better/num_samples))
    out = {"new_pvalue": 1 - (new_better/num_samples)}
    print()
    return out


def bootstrap_bleu(references, baseline_hypo, new_hypo, num_samples=3000, sample_size=200):
    """
    Original Code adapted from https://github.com/pytorch/translate/tree/master/pytorch_translate
    Attributes:
        references -> the reference data
        baseline_hypo -> the baseline model output
        new_hypo -> new mdoel_hypo
    """

    import numpy as np
    name_list = ['bleu1', 'bleu2', 'bleu3', 'bleu4']
    num_sents = len(baseline_hypo)
    assert len(baseline_hypo) == len(new_hypo) == len(references)

    indices = np.random.randint(
        low=0, high=num_sents, size=(num_samples, sample_size))
    out = {}
    baseline_better = [0, 0, 0, 0]
    new_better = [0, 0, 0, 0]
    num_equal = [0, 0, 0, 0]

    for index in indices:

        sub_base_hypo = [y for i, y in enumerate(
            baseline_hypo) if i in index]
        sub_new_hypo = [y for i, y in enumerate(
            new_hypo) if i in index]
        sub_ref = [y for i, y in enumerate(
            references) if i in index]

        baseline_bleu = sacrebleu.raw_corpus_bleu(
            sys_stream=sub_base_hypo,
            ref_streams=[sub_ref]
        ).scores

        new_bleu = sacrebleu.raw_corpus_bleu(
            sys_stream=sub_new_hypo,
            ref_streams=[sub_ref]
        ).scores

        for score_name in range(4):
            if new_bleu[score_name] > baseline_bleu[score_name]:
                new_better[score_name] += 1

            elif new_bleu[score_name] < baseline_bleu[score_name]:
                baseline_better[score_name] += 1
            else:
                num_equal[score_name] += 1

    print("------DOWN BLEU COMPUTATION------")
    for score_num in range(4):
        out["BLEU-" + str(score_num+1)] = {
            "new_pvalue": 1 - (new_better[score_num]/num_samples), "old_p_value": 1 - (baseline_better[score_num]/num_samples)}
        print("BLEU {} baseline better confidence : {:.2f}".format(
            score_num+1, baseline_better[score_num]/num_samples))
        print("BLEU {} New better confidence : {:.2f}".format(
            score_num+1, new_better[score_num]/num_samples))
        print()
    return out


def get_alignment(r, h, d):
    """
    Original Code from https://github.com/zszyellow/WER-in-python/blob/master/wer.py
    This function is to get the list of steps in the process of dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calculating the editing distance of h and r.
    """
    x = len(r)
    y = len(h)
    max_len = 3 * (x + y)

    alignlist = []
    align_ref = ""
    align_hyp = ""
    alignment = ""

    while True:
        if (x <= 0 and y <= 0) or (len(alignlist) > max_len):
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] and r[x - 1] == h[y - 1]:
            align_hyp = " " + h[y - 1] + align_hyp
            align_ref = " " + r[x - 1] + align_ref
            alignment = " " * (len(r[x - 1]) + 1) + alignment
            alignlist.append("C")
            x = max(x - 1, 0)
            y = max(y - 1, 0)
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] + WER_COST_SUB:
            ml = max(len(h[y - 1]), len(r[x - 1]))
            align_hyp = " " + h[y - 1].ljust(ml) + align_hyp
            align_ref = " " + r[x - 1].ljust(ml) + align_ref
            alignment = " " + "S" + " " * (ml - 1) + alignment
            alignlist.append("S")
            x = max(x - 1, 0)
            y = max(y - 1, 0)
        elif y >= 1 and d[x][y] == d[x][y - 1] + WER_COST_INS:
            align_hyp = " " + h[y - 1] + align_hyp
            align_ref = " " + "*" * len(h[y - 1]) + align_ref
            alignment = " " + "I" + " " * (len(h[y - 1]) - 1) + alignment
            alignlist.append("I")
            x = max(x, 0)
            y = max(y - 1, 0)
        else:
            align_hyp = " " + "*" * len(r[x - 1]) + align_hyp
            align_ref = " " + r[x - 1] + align_ref
            alignment = " " + "D" + " " * (len(r[x - 1]) - 1) + alignment
            alignlist.append("D")
            x = max(x - 1, 0)
            y = max(y, 0)

    align_ref = align_ref[1:]
    align_hyp = align_hyp[1:]
    alignment = alignment[1:]

    return (
        alignlist[::-1],
        {"align_ref": align_ref, "align_hyp": align_hyp, "alignment": alignment},
    )
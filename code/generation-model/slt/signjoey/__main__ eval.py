from bert_score import score
import argparse
import os
import sys
from signjoey.metrics import bleu, chrf, rouge, bootstrap_bleu, bootstrap_bertscore, bootstrap_rogue
import pickle

# coding: utf-8
"""
This module holds evaluation method for MT results.
"""

# path


def eval_score(args):
    print("EVAL ON ...", args.model_name)
    # dev
    dev_path = "output/%s.dev.txt" % args.model_name
    dev_file = "output/dev.text"
    # test
    test_path = "output/%s.test.txt" % args.model_name
    test_file = "output/test.text"

    # read both files.
    dev_hypo = [y.split("|")[1].strip()
                for y in open(dev_path, "r").readlines()]
    test_hypo = [y.split("|")[1].strip()
                 for y in open(test_path, "r").readlines()]
    dev_gold = [y.strip() for y in open(dev_file, "r").readlines()]
    test_gold = [y.strip() for y in open(test_file, "r").readlines()]

    # load the marked indexes.
    print("Labeled Dev")
    dev_marked = pickle.load(open("output/dev_positive_labels.p", "rb"))
    test_marked = pickle.load(open("output/test_positive_labels.p", "rb"))
    dev_hypo_marked = [y for i, y in enumerate(dev_hypo) if i in dev_marked]
    dev_gold_marked = [y for i, y in enumerate(dev_gold) if i in dev_marked]
    dev_hypo_neg = [y for i, y in enumerate(dev_hypo) if i not in dev_marked]
    dev_gold_neg = [y for i, y in enumerate(dev_gold) if i not in dev_marked]

    test_hypo_marked = [y for i, y in enumerate(test_hypo) if i in test_marked]
    test_gold_marked = [y for i, y in enumerate(test_gold) if i in test_marked]
    test_hypo_neg = [y for i, y in enumerate(
        test_hypo) if i not in test_marked]
    test_gold_neg = [y for i, y in enumerate(
        test_gold) if i not in test_marked]

    print("All DEV result ...")
    txt_bleu = bleu(references=dev_gold, hypotheses=dev_hypo)
    txt_chrf = chrf(references=dev_gold, hypotheses=dev_hypo)
    txt_rouge = rouge(references=dev_gold, hypotheses=dev_hypo)
    P, R, F1 = score(dev_hypo, dev_gold, lang='others')
    print(txt_bleu)
    print(txt_rouge)
    print(P.mean(), R.mean(), F1.mean())

    print("All TEST result ...")
    txt_bleu = bleu(references=test_gold, hypotheses=test_hypo)
    txt_chrf = chrf(references=test_gold, hypotheses=test_hypo)
    txt_rouge = rouge(references=test_gold, hypotheses=test_hypo)
    P, R, F1 = score(test_hypo, test_gold, lang='others')
    print(txt_bleu)
    print(txt_rouge)
    print(P.mean(), R.mean(), F1.mean())

    # prediction
    print("DEV RESULT with labels")
    txt_bleu = bleu(references=dev_gold_marked, hypotheses=dev_hypo_marked)
    txt_chrf = chrf(references=dev_gold_marked, hypotheses=dev_hypo_marked)
    txt_rouge = rouge(references=dev_gold_marked, hypotheses=dev_hypo_marked)
    P, R, F1 = score(dev_hypo_marked, dev_gold_marked, lang='others')
    print(txt_bleu)
    print(txt_rouge)
    print(P.mean(), R.mean(), F1.mean())

    print("DEV RESULT without labels")
    txt_bleu = bleu(references=dev_gold_neg, hypotheses=dev_hypo_neg)
    txt_chrf = chrf(references=dev_gold_neg, hypotheses=dev_hypo_neg)
    txt_rouge = rouge(references=dev_gold_neg, hypotheses=dev_hypo_neg)
    P, R, F1 = score(dev_hypo_neg, dev_gold_neg, lang='others')
    print(txt_bleu)
    print(txt_rouge)
    print(P.mean(), R.mean(), F1.mean())

    print("TEST RESULT with labels")
    txt_bleu = bleu(references=test_gold_marked, hypotheses=test_hypo_marked)
    txt_chrf = chrf(references=test_gold_marked, hypotheses=test_hypo_marked)
    txt_rouge = rouge(references=test_gold_marked, hypotheses=test_hypo_marked)
    P, R, F1 = score(test_hypo_marked, test_gold_marked, lang='others')
    print(txt_bleu)
    print(txt_rouge)
    print(P.mean(), R.mean(), F1.mean())

    print("TEST RESULT with out labels")
    txt_bleu = bleu(references=test_gold_neg, hypotheses=test_hypo_neg)
    txt_chrf = chrf(references=test_gold_neg, hypotheses=test_hypo_neg)
    txt_rouge = rouge(references=test_gold_neg, hypotheses=test_hypo_neg)
    P, R, F1 = score(test_hypo_neg, test_gold_neg, lang='others')
    print(txt_bleu)
    print(txt_rouge)
    print(P.mean(), R.mean(), F1.mean())

# paired test for significance test.


def eval_paired_score(args):
    print("EVAL ON ...", args.model_name)
    # dev
    dev_path = "output/%s.dev.txt" % args.model_name
    dev_file = "output/dev.text"
    dev_baseline_file = "output/train_gloss_default.BW_06.A_3.dev.txt"
    # test
    test_path = "output/%s.test.txt" % args.model_name
    test_file = "output/test.text"
    test_baseline_file = "output/train_gloss_default.BW_06.A_3.test.txt"

    # read both files.
    dev_hypo = [y.split("|")[1].strip()
                for y in open(dev_path, "r").readlines()]
    test_hypo = [y.split("|")[1].strip()
                 for y in open(test_path, "r").readlines()]
    dev_base_hypo = [y.split("|")[1].strip()
                     for y in open(dev_baseline_file, "r").readlines()]
    dev_gold = [y.strip() for y in open(dev_file, "r").readlines()]
    test_gold = [y.strip() for y in open(test_file, "r").readlines()]
    test_base_hypo = [y.split("|")[1].strip()
                      for y in open(test_baseline_file, "r").readlines()]

    # load the marked indexes.
    print("Labeled Dev")
    dev_marked = pickle.load(open("output/dev_positive_labels.p", "rb"))
    test_marked = pickle.load(open("output/test_positive_labels.p", "rb"))
    dev_hypo_marked = [y for i, y in enumerate(dev_hypo) if i in dev_marked]
    dev_gold_marked = [y for i, y in enumerate(dev_gold) if i in dev_marked]
    dev_baseline_marked = [y for i, y in enumerate(
        dev_base_hypo) if i in dev_marked]
    dev_hypo_neg = [y for i, y in enumerate(dev_hypo) if i not in dev_marked]
    dev_gold_neg = [y for i, y in enumerate(dev_gold) if i not in dev_marked]
    dev_baseline_neg = [y for i, y in enumerate(
        dev_base_hypo) if i not in dev_marked]

    test_hypo_marked = [y for i, y in enumerate(test_hypo) if i in test_marked]
    test_gold_marked = [y for i, y in enumerate(test_gold) if i in test_marked]
    test_baseline_marked = [y for i, y in enumerate(
        test_base_hypo) if i in test_marked]
    test_hypo_neg = [y for i, y in enumerate(
        test_hypo) if i not in test_marked]
    test_gold_neg = [y for i, y in enumerate(
        test_gold) if i not in test_marked]
    test_baseline_neg = [y for i, y in enumerate(
        test_base_hypo) if i not in test_marked]

    all_ref = [dev_gold, test_gold, dev_gold_marked,
               dev_gold_neg, test_gold_marked, test_gold_neg]
    all_hypo = [dev_hypo, test_hypo, dev_hypo_marked,
                dev_hypo_neg, test_hypo_marked, test_hypo_neg]
    all_baseline = [dev_base_hypo, test_base_hypo, dev_baseline_marked,
                    dev_baseline_neg, test_baseline_marked, test_baseline_neg]
    name = ['All DEV', "All Test", "All Dev marked",
            "All Dev neg", "All Test marked", "All Test neg"]

    for i in range(len(all_ref)):
        print(name[i], "#######################")
        ref_, hypo_, baseline_ = all_ref[i], all_hypo[i], all_baseline[i]
        paired_bleu_out = bootstrap_bleu(ref_, baseline_, hypo_)
        paired_rouge_out = bootstrap_rogue(ref_, baseline_, hypo_)
        P, R, baseline_F1 = score(baseline_, ref_, lang='others')
        P, R, new_F1 = score(hypo_, ref_, lang='others')
        paired_bertscore = bootstrap_bertscore(baseline_F1, new_F1)
        print()


def eval_adj(args):
    dev_path = "output/%s.dev.txt" % args.model_name
    dev_file = "output/dev.text"
    # test
    test_path = "output/%s.test.txt" % args.model_name
    test_file = "output/test.text"

    # read both files.
    dev_hypo = [y.split("|")[1].strip()
                for y in open(dev_path, "r").readlines()]
    test_hypo = [y.split("|")[1].strip()
                 for y in open(test_path, "r").readlines()]
    dev_gold = [y.strip() for y in open(dev_file, "r").readlines()]
    test_gold = [y.strip() for y in open(test_file, "r").readlines()]

    # load the marked indexes.
    print("Labeled Dev")
    dev_marked = pickle.load(open("output/dev_positive_labels.p", "rb"))
    test_marked = pickle.load(open("output/test_positive_labels.p", "rb"))
    dev_hypo_marked = [y for i, y in enumerate(dev_hypo) if i in dev_marked]
    dev_gold_marked = [y for i, y in enumerate(dev_gold) if i in dev_marked]
    dev_hypo_neg = [y for i, y in enumerate(dev_hypo) if i not in dev_marked]
    dev_gold_neg = [y for i, y in enumerate(dev_gold) if i not in dev_marked]

    test_hypo_marked = [y for i, y in enumerate(test_hypo) if i in test_marked]
    test_gold_marked = [y for i, y in enumerate(test_gold) if i in test_marked]
    test_hypo_neg = [y for i, y in enumerate(
        test_hypo) if i not in test_marked]
    test_gold_neg = [y for i, y in enumerate(
        test_gold) if i not in test_marked]

    # import spacy.
    import spacy
    nlp = spacy.load('de_core_news_md')

    # DEV
    dev_token_hypo = []
    dev_token_gold = []
    test_token_hypo = []
    test_token_gold = []

    for sent in dev_hypo_marked:
        doc = nlp(sent)
        hypo = []
        for token in doc:
            if "ADJ" in str(token.pos_) or "ADV" in str(token.pos_):
                hypo.append(str(token.pos_))
        dev_token_hypo.append(hypo)

    for sent in dev_gold_marked:
        doc = nlp(sent)
        gold = []
        for token in doc:
            if "ADJ" in str(token.pos_) or "ADV" in str(token.pos_):
                gold.append(str(token.pos_))

        dev_token_gold.append(gold)
    print(dev_token_hypo[0])
    print(dev_token_gold[0])
    for sent in test_hypo_marked:
        doc = nlp(sent)
        hypo = []
        for token in doc:
            if "ADJ" in str(token.pos_) or "ADV" in str(token.pos_):
                hypo.append(str(token.pos_))
        test_token_hypo.append(hypo)
    for sent in test_gold_marked:
        doc = nlp(sent)
        gold = []
        for token in doc:
            if "ADJ" in str(token.pos_) or "ADV" in str(token.pos_):
                gold.append(str(token.pos_))

        test_token_gold.append(gold)
    print(test_token_hypo[0])
    print(test_token_gold[0])
    pickle.dump(dev_token_hypo, open("%s_dev_hypo.p" % args.model_name, "wb"))
    pickle.dump(test_token_hypo, open(
        "%s_test_hypo.p" % args.model_name, "wb"))
    pickle.dump(dev_token_gold, open("%s_dev_gold.p" % args.model_name, "wb"))
    pickle.dump(test_token_gold, open(
        "%s_test_gold.p" % args.model_name, "wb"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    args = parser.parse_args()

    eval_score(args)
    eval_paired_score(args)
    # eval_adj(args)


main()

"""
Part 4: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""

import math
from collections import defaultdict, Counter
from itertools import chain

epsilon_for_pt = 1e-6
emit_epsilon = 1e-6
laplace_constant = 1e-6
beam_width = 5

def map_to_pseudoword(word):
    mapping_rules = {
        "ly": "X-LY",
        "ing": "X-ING",
        "ed": "X-ED",
        "s": "X-S",
        "ment": "X-MENT",
        "ous": "X-OUS",
        "ive": "X-IVE",
        "tion": "X-TION",
        "ness": "X-NESS",
        "able": "X-ABLE",
        "ful": "X-FUL",
        "th": "X-TH",
        "teen": "X-TEEN",
        "ty": "X-TY",
        "and": "X-AND",
        "or": "X-OR",
        "but": "X-BUT",
        "al": "X-AL",
        "ic": "X-IC"
    }

    for pattern, pseudoword in mapping_rules.items():
        if word.endswith(pattern):
            return pseudoword

    return "UNKNOWN"

def training(sentences):
    init_prob = defaultdict(float)
    emit_prob = defaultdict(lambda: defaultdict(float))
    trans_prob = defaultdict(lambda: defaultdict(float))
    hapax_word_tags = defaultdict(float)

    word_counts = Counter(word for word, _ in chain(*sentences))
    hapax_words = {word for word, count in word_counts.items() if count == 1}

    for sentence in sentences:
        first_tag = sentence[0][1]
        init_prob[first_tag] += 1

    total_sentences = len(sentences)
    for tag in init_prob:
        init_prob[tag] /= total_sentences

    total_tags = Counter(tag for _, tag in chain(*sentences))
    
    pseudoword_emit_counts = defaultdict(lambda: defaultdict(float))
    pseudoword_pattern_counts = defaultdict(float)

    for sentence in sentences:
        for i, (word, tag) in enumerate(sentence):
            emit_prob[tag][word] += 1.5

            if word in hapax_words:
                hapax_word_tags[tag] += 1
                pseudoword = map_to_pseudoword(word)
                if pseudoword != "UNKNOWN":
                    pseudoword_emit_counts[tag][pseudoword] += 1.5
                    pseudoword_pattern_counts[pseudoword] += 1

            if i < len(sentence) - 1:
                next_tag = sentence[i + 1][1]
                trans_prob[tag][next_tag] += 1.5

    total_pseudoword_count = sum(pseudoword_pattern_counts.values())
    pseudoword_weights = {pseudoword: count/total_pseudoword_count for pseudoword, count in pseudoword_pattern_counts.items()}

    for tag, pseudoword_counts in pseudoword_emit_counts.items():
        for pseudoword, count in pseudoword_counts.items():
            weight = pseudoword_weights.get(pseudoword, 1)
            if pseudoword not in emit_prob[tag]:
                emit_prob[tag][pseudoword] = weight * 1.5
            emit_prob[tag][pseudoword] += count

    hapax_sum = sum(hapax_word_tags.values())
    for tag in hapax_word_tags:
        hapax_word_tags[tag] /= hapax_sum

    for tag, word_counts in emit_prob.items():
        total_count = sum(word_counts.values())
        for word, count in word_counts.items():
            emit_prob[tag][word] = count / total_count

    for tag, next_tag_counts in trans_prob.items():
        for next_tag, count in next_tag_counts.items():
            trans_prob[tag][next_tag] = count / total_tags[tag]

    return init_prob, emit_prob, trans_prob, hapax_word_tags

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob, total_tags, hapax_word_tags):
    log_prob = {}
    predict_tag_seq = {}

    for current_tag in emit_prob.keys():
        max_prob = float('-inf')
        max_tag = None

        top_prev_tags = sorted(prev_prob, key=prev_prob.get, reverse=True)[:beam_width]

        for prev_tag in top_prev_tags:
            if word in emit_prob[current_tag]:
                emission = emit_prob[current_tag][word]
            else:
                pseudoword = map_to_pseudoword(word)
                alpha = laplace_constant * hapax_word_tags.get(current_tag, 0.00001)
                emission = alpha / (total_tags[current_tag] + alpha * len(emit_prob[current_tag]))
                if pseudoword in emit_prob[current_tag]:
                    emission *= emit_prob[current_tag][pseudoword]
                else:
                    emission *= emit_epsilon

            if current_tag in trans_prob[prev_tag]:
                transition = trans_prob[prev_tag][current_tag]
            else:
                transition = epsilon_for_pt / total_tags[prev_tag]

            prob = prev_prob[prev_tag] + math.log(transition) + math.log(emission)
            if prob > max_prob:
                max_prob = prob
                max_tag = prev_tag

        log_prob[current_tag] = max_prob
        predict_tag_seq[current_tag] = prev_predict_tag_seq[max_tag] + [current_tag]

    return log_prob, predict_tag_seq


def viterbi_3(train, test, get_probs=training):
    init_prob, emit_prob, trans_prob, hapax_word_tags = get_probs(train)
    total_tags = Counter(tag for _, tag in chain(*train))
    predicts = []

    for sentence in test:
        try:
            words = [word for word, _ in sentence]
        except ValueError:
            words = sentence

        log_prob = {tag: math.log(init_prob[tag] + epsilon_for_pt) + math.log(emit_prob[tag].get(words[0], emit_epsilon)) for tag in total_tags}
        predict_tag_seq = {tag: [tag] for tag in total_tags}

        for i in range(1, len(words)):
            log_prob, predict_tag_seq = viterbi_stepforward(i, words[i], log_prob, predict_tag_seq, emit_prob, trans_prob, total_tags, hapax_word_tags)

        final_tag = max(log_prob, key=log_prob.get)
        best_tag_seq = predict_tag_seq[final_tag]

        predicts.append(list(zip(words, best_tag_seq)))

    return predicts

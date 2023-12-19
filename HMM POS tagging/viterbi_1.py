#Kalaipriyan R

import math
from collections import defaultdict, Counter
from itertools import chain

epsilon_for_pt = 1e-6
emit_epsilon = 1e-6

def training(sentences):
    init_prob = defaultdict(float)
    emit_prob = defaultdict(lambda: defaultdict(float))
    trans_prob = defaultdict(lambda: defaultdict(float))

    for sentence in sentences:
        first_tag = sentence[0][1]
        init_prob[first_tag] += 1.0
    total_sentences = len(sentences)
    for tag in init_prob:
        init_prob[tag] /= total_sentences

    total_tags = Counter(tag for _, tag in chain(*sentences))
    for sentence in sentences:
        for i, (word, tag) in enumerate(sentence):
            emit_prob[tag][word] += 1.0
            if i < len(sentence) - 1:
                next_tag = sentence[i + 1][1]
                trans_prob[tag][next_tag] += 1.0

    for tag, word_counts in emit_prob.items():
        for word, count in word_counts.items():
            emit_prob[tag][word] = count / total_tags[tag]
    
    for tag, next_tag_counts in trans_prob.items():
        for next_tag, count in next_tag_counts.items():
            trans_prob[tag][next_tag] = count / total_tags[tag]

    return init_prob, emit_prob, trans_prob

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob, total_tags):
    log_prob = {}
    predict_tag_seq = {}

    for current_tag in emit_prob.keys():
        max_prob = float('-inf')
        max_tag = None

        for prev_tag in prev_prob.keys():
            if word in emit_prob[current_tag]:
                emission = emit_prob[current_tag][word]
            else:
                emission = emit_epsilon / total_tags[current_tag]
            
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

def viterbi_1(train, test, get_probs=training):
    init_prob, emit_prob, trans_prob = get_probs(train)
    total_tags = Counter(tag for _, tag in chain(*train))
    predicts = []

    for sentence in test:
        log_prob = {tag: math.log(init_prob[tag]) if tag in init_prob else math.log(epsilon_for_pt) for tag in emit_prob.keys()}
        predict_tag_seq = {tag: [tag] for tag in emit_prob.keys()}

        for i in range(1, len(sentence)):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob, trans_prob, total_tags)

        final_tag = max(log_prob, key=log_prob.get)
        predicts.append(list(zip(sentence, predict_tag_seq[final_tag])))

    return predicts

# -*- coding: utf-8 -*-

import json
from util import xavier_vector, normalise_word_vectors
from model import model_definition, max_utterance_length, vector_dimension, max_no_turns
import tensorflow as tf
import numpy as np
import sys
import os
import time
from copy import deepcopy
import math
import click
from collections import OrderedDict
from random import shuffle

VALIDATION_URL = "data/validate.json"
WORD_VECTORS_URL = "word-vectors/paragram_300_sl999.txt"
TRAINING_URL = "data/train.json"
ONTOLOGY_URL = "data/ontology.json"
TESTING_URL = "data/test.json"
MODEL_URL = "models/model-1"
GRAPH_URL = "graphs/graph-1"
RESULTS_URL = "results/log-1.txt"
domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi']

train_batch_size = 64
batches_per_eval = 10
no_epochs = 600
device = "gpu"
start_batch = 0

num_slots = 0

booking_slots = {}


def load_ontology(url, word_vectors):
    '''
    Load the ontology from a file
    :param url: to the ontology
    :param word_vectors: dictionary of the word embeddings [words, vector_dimension]
    :return: list([domain-slot-value]), [no_slots, vector_dimension]
    '''
    global num_slots
    print("Loading the ontology....................")
    data = json.load(open(url, mode='r', encoding='utf8'), object_pairs_hook=OrderedDict)
    slot_values = []
    ontology = []
    slots_values = []
    ontology_vectors = []
    for slots in data:
        [domain, slot] = slots.split('-')
        if domain not in domains or slot == 'name':
            continue
        values = data[slots]
        if "book" in slot:
            [slot, value] = slot.split(" ")
            booking_slots[domain+'-'+value] = values
            values = [value]
        elif slot == "departure" or slot == "destination":
            values = ["place"]
        domain_vec = np.sum(process_text(domain, word_vectors), axis=0)
        if domain not in word_vectors:
            word_vectors[domain.replace(" ", "")] = domain_vec
        slot_vec = np.sum(process_text(slot, word_vectors), axis=0)
        if domain+'-'+slot not in slots_values:
            slots_values.append(domain+'-'+slot)
        if slot not in word_vectors:
            word_vectors[slot.replace(" ", "")] = slot_vec
        slot_values.append(len(values))
        for value in values:
            ontology.append(domain + '-' + slot + '-' + value)
            value_vec = np.sum(process_text(value, word_vectors, print_mode=True), axis=0)
            if value not in word_vectors:
                word_vectors[value.replace(" ", "")] = value_vec
            ontology_vectors.append(np.concatenate((domain_vec, slot_vec, value_vec)))

    num_slots = len(slots_values)
    print("We have about {} values".format(len(ontology)))
    print("The Full Ontology is:")
    print(ontology)
    print("The slots in this ontology:")
    print(slots_values)
    return ontology, np.asarray(ontology_vectors, dtype='float32'), slot_values


def load_word_vectors(url):
    '''
    Load the word embeddings from the url
    :param url: to the word vectors
    :return: dict of word and vector values
    '''
    word_vectors = {}
    print("Loading the word embeddings....................")
    with open(url, mode='r', encoding='utf8') as f:
        for line in f:
            line = line.split(" ", 1)
            key = line[0]
            word_vectors[key] = np.fromstring(line[1], dtype="float32", sep=" ")
    print("The vocabulary contains about {} word embeddings".format(len(word_vectors)))
    return normalise_word_vectors(word_vectors)


def track_dialogue(data, ontology, predictions, y):
    overall_accuracy_total = 0
    overall_accuracy_corr = 0
    joint_accuracy_total = 0
    joint_accuracy_corr = 0
    global num_slots
    dialogues = []
    idx = 0
    for dialogue in data:
        turn_ids = []
        for key in dialogue.keys():
            if key.isdigit():
                turn_ids.append(int(key))
            elif dialogue[key] and key not in domains:
                continue
        turn_ids.sort()
        turns = []
        previous_terms = []
        for key in turn_ids:
            turn = dialogue[str(key)]
            user_input = turn['user']['text']
            sys_res = turn['system']
            state = turn['user']['belief_state']
            turn_obj = dict()
            turn_obj['user'] = user_input
            turn_obj['system'] = sys_res
            prediction = predictions[idx, :]
            indices = np.argsort(prediction)[:-(int(np.sum(prediction)) + 1):-1]
            predicted_terms = [process_booking(ontology[i], user_input, previous_terms) for i in indices]
            previous_terms = deepcopy(predicted_terms)
            turn_obj['prediction'] = ["{}: {}".format(predicted_terms[x], y[idx, i]) for x, i in enumerate(indices)]
            turn_obj['True state'] = []
            idx += 1
            unpredicted_labels = 0
            for domain in state:
                if domain not in domains:
                    continue
                slots = state[domain]['semi']
                for slot in slots:
                    if slot == 'name':
                        continue
                    value = slots[slot]
                    if value != '':
                        label = domain + '-' + slot + '-' + value
                        turn_obj['True state'].append(label)
                        if label in predicted_terms:
                            predicted_terms.remove(label)
                        else:
                            unpredicted_labels += 1

            turns.append(turn_obj)
            overall_accuracy_total += num_slots
            overall_accuracy_corr += (num_slots - unpredicted_labels - len(predicted_terms))
            if unpredicted_labels + len(predicted_terms) == 0:
                joint_accuracy_corr += 1
            joint_accuracy_total += 1

        dialogues.append(turns)
    return dialogues, overall_accuracy_corr/overall_accuracy_total, joint_accuracy_corr/joint_accuracy_total


def process_booking(ontolog_term, usr_input, previous_terms):
    usr_input = usr_input.lower().split()
    domain, slot, value = ontolog_term.split('-')
    if slot == 'book':
        for term in previous_terms:
            if domain+'-book '+value in term:
                ontolog_term = term
                break
        else:
            if value == 'stay' or value == 'people':
                numbers = [int(s) for s in usr_input if s.isdigit()]
                if len(numbers) == 1:
                    ontolog_term = domain + '-' + slot + ' ' + value + '-' + str(numbers[0])
                elif len(numbers) == 2:
                    vals = {}
                    if usr_input[usr_input.index(str(numbers[0]))+1] in ['people', 'person']:
                        vals['people'] = str(numbers[0])
                        vals['stay'] = str(numbers[1])
                    else:
                        vals['people'] = str(numbers[1])
                        vals['stay'] = str(numbers[0])
                    ontolog_term = domain + '-' + slot + ' ' + value + '-' + vals[value]
            else:
                for val in booking_slots[domain+'-'+value]:
                    if val in ' '.join(usr_input):
                        ontolog_term = domain + '-' + slot + ' ' + value + '-' + val
                        break
    return ontolog_term


def load_woz_data(data, word_vectors, ontology, url=True):
    '''
    Load the woz3 data and extract feature vectors
    :param data: the data to load
    :param word_vectors: word embeddings
    :param ontology: list of domain-slot-value
    :param url: Is the data coming from a url, default true
    :return: list(num of turns, user_input vectors, system_response vectors, labels)
    '''
    if url:
        print("Loading data from url {} ....................".format(data))
        data = json.load(open(data, mode='r', encoding='utf8'))

    dialogues = []
    actual_dialogues = []
    for dialogue in data:
        turn_ids = []
        for key in dialogue.keys():
            if key.isdigit():
                turn_ids.append(int(key))
            elif dialogue[key] and key not in domains:
                continue
        turn_ids.sort()
        num_turns = len(turn_ids)
        user_vecs = []
        sys_vecs = []
        turn_labels = []
        turn_domain_labels = []
        add = False
        good = True
        pre_sys = np.zeros([max_utterance_length, vector_dimension], dtype="float32")
        for key in turn_ids:
            turn = dialogue[str(key)]
            user_v, sys_v, labels, domain_labels = process_turn(turn, word_vectors, ontology)
            if good and (user_v.shape[0] > max_utterance_length or pre_sys.shape[0] > max_utterance_length):
                good = False
                break
            user_vecs.append(user_v)
            sys_vecs.append(pre_sys)
            turn_labels.append(labels)
            turn_domain_labels.append(domain_labels)
            if not add and sum(labels) > 0:
                add = True
            pre_sys = sys_v
        if add and good:
            dialogues.append((num_turns, user_vecs, sys_vecs, turn_labels, turn_domain_labels))
            actual_dialogues.append(dialogue)
    print("The data contains about {} dialogues".format(len(dialogues)))
    return dialogues, actual_dialogues


def process_turn(turn, word_vectors, ontology):
    '''
    Process a single turn extracting and processing user text, system response and labels
    :param turn: dict
    :param word_vectors: word embeddings
    :param ontology: list(domain-slot-value)
    :return: ([utterance length, 300], [utterance length, 300], [no_slots])
    '''
    user_input = turn['user']['text']
    sys_res = turn['system']
    state = turn['user']['belief_state']
    user_v = process_text(user_input, word_vectors, ontology)
    sys_v = process_text(sys_res, word_vectors, ontology)
    labels = np.zeros(len(ontology), dtype='float32')
    domain_labels = np.zeros(len(ontology), dtype='float32')
    for domain in state:
        if domain not in domains:
            continue
        slots = state[domain]['semi']
        domain_mention = False
        for slot in slots:

            if slot == 'name':
                continue
            value = slots[slot]
            if "book" in slot:
                [slot, value] = slot.split(" ")
            if value != '' and value != 'corsican':
                if slot == "destination" or slot == "departure":
                    value = "place"
                elif value == '09;45':
                    value = '09:45'
                elif 'alpha-milton' in value:
                    value = value.replace('alpha-milton', 'alpha milton')
                elif value == 'east side':
                    value = 'east'
                elif value == ' expensive':
                    value = 'expensive'
                labels[ontology.index(domain + '-' + slot + '-' + value)] = 1
                domain_mention = True
        if domain_mention:
            for idx, slot in enumerate(ontology):
                if domain in slot:
                    domain_labels[idx] = 1

    return user_v, sys_v, labels, domain_labels


def process_text(text, word_vectors, ontology=None, print_mode=False):
    '''
    Process a line/sentence converting words to feature vectors
    :param text: sentence
    :param word_vectors: word embeddings
    :param ontology: The ontology to do exact matching
    :param print_mode: Log the cases where the word is not in the pre-trained word vectors
    :return: [length of sentence, 300]
    '''
    text = text.replace("(", "").replace(")", "").replace('"', "").replace(u"’", "'").replace(u"‘", "'")
    text = text.replace("\t", "").replace("\n", "").replace("\r", "").strip().lower()
    text = text.replace(',', ' ').replace('.', ' ').replace('?', ' ').replace('-', ' ').replace('/', ' / ')\
        .replace(':', ' ')
    if ontology:
        for slot in ontology:
            [domain, slot, value] = slot.split('-')
            text.replace(domain, domain.replace(" ", ""))\
                .replace(slot, slot.replace(" ", ""))\
                .replace(value, value.replace(" ", ""))

    words = text.split()

    vectors = []
    for word in words:
        word = word.replace("'", "").replace("!", "")
        if word == "":
            continue
        if word not in word_vectors:
            length = len(word)
            for i in range(1, length)[::-1]:
                if word[:i] in word_vectors and word[i:] in word_vectors:
                    vec = word_vectors[word[:i]] + word_vectors[word[i:]]
                    break
            else:
                vec = xavier_vector(word)
                word_vectors[word] = vec
                if print_mode:
                    print("Adding new word: {}".format(word))
        else:
            vec = word_vectors[word]
        vectors.append(vec)
    return np.asarray(vectors, dtype='float32')


def generate_batch(dialogues, batch_no, batch_size, ontology_size):
    '''
    Generate examples for minibatch training
    :param dialogues: list(num of turns, user_input vectors, system_response vectors, labels)
    :param batch_no: where we are in the training data
    :param batch_size: number of dialogues to generate
    :param ontology_size: no_slots
    :return: list(user_input, system_response, labels, user_sentence_length, system_sentence_length, number of turns)
    '''
    user = np.zeros((batch_size, max_no_turns, max_utterance_length, vector_dimension), dtype='float32')
    sys_res = np.zeros((batch_size, max_no_turns, max_utterance_length, vector_dimension), dtype='float32')
    labels = np.zeros((batch_size, max_no_turns, ontology_size), dtype='float32')
    domain_labels = np.zeros((batch_size, max_no_turns, ontology_size), dtype='float32')
    user_uttr_len = np.zeros((batch_size, max_no_turns), dtype='int32')
    sys_uttr_len = np.zeros((batch_size, max_no_turns), dtype='int32')
    no_turns = np.zeros(batch_size, dtype='int32')
    idx = 0
    for i in range(batch_no*train_batch_size, batch_no*train_batch_size + batch_size):
        (num_turns, user_vecs, sys_vecs, turn_labels, turn_domain_labels) = dialogues[i]
        no_turns[idx] = num_turns
        for j in range(num_turns):
            user_uttr_len[idx, j] = user_vecs[j].shape[0]
            sys_uttr_len[idx, j] = sys_vecs[j].shape[0]
            user[idx, j, :user_uttr_len[idx, j], :] = user_vecs[j]
            sys_res[idx, j, :sys_uttr_len[idx, j], :] = sys_vecs[j]
            labels[idx, j, :] = turn_labels[j]
            domain_labels[idx, j, :] = turn_domain_labels[j]
        idx += 1
    return user, sys_res, labels, domain_labels, user_uttr_len, sys_uttr_len, no_turns


def evaluate_model(sess, model_variables, val_data, summary, batch_id, i):

    '''
    Evaluate the model against validation set
    :param sess: training session
    :param model_variables: all model input variables
    :param val_data: validation data
    :param summary: For tensorboard
    :param batch_id: where we are in the training data
    :param i: the index of the validation data to load
    :return: evaluation accuracy and the summary
    '''

    (user, sys_res, no_turns, user_uttr_len, sys_uttr_len, labels, domain_labels, domain_accuracy,
     slot_accuracy, value_accuracy, value_f1, train_step, keep_prob, _, _, _) = model_variables

    batch_user, batch_sys, batch_labels, batch_domain_labels, batch_user_uttr_len, batch_sys_uttr_len, \
        batch_no_turns = val_data

    start_time = time.time()

    b_z = train_batch_size
    [precision, recall, value_f1] = value_f1
    [d_acc, s_acc, v_acc, f1_score, pr, re, sm1, sm2] = sess.run([domain_accuracy, slot_accuracy, value_accuracy,
                                                                  value_f1, precision, recall] + summary,
                                                           feed_dict={user: batch_user[i:i+b_z, :, :, :],
                                                                      sys_res: batch_sys[i:i+b_z, :, :, :],
                                                                      labels: batch_labels[i:i+b_z, :, :],
                                                                      domain_labels: batch_domain_labels[i:i+b_z, :, :],
                                                                      user_uttr_len: batch_user_uttr_len[i:i+b_z, :],
                                                                      sys_uttr_len: batch_sys_uttr_len[i:i+b_z, :],
                                                                      no_turns: batch_no_turns[i:i+b_z],
                                                                      keep_prob: 1.0})

    print("Batch", batch_id, "[Domain Accuracy] = ", d_acc, "[Slot Accuracy] = ", s_acc, "[Value Accuracy] = ",
          v_acc, "[F1 Score] = ", f1_score, "[Precision] = ", pr, "[Recall] = ", re,
          " ----- ", round(time.time() - start_time, 3),
          "seconds. ---")

    return d_acc, s_acc, v_acc, f1_score, sm1, sm2


@click.group()
def cli():
    pass


@cli.command()
@click.option('--num_hid', type=int)
@click.option('--bidir/--no-bidir', default=True)
@click.option('--net_type')
@click.option('--n2p', type=int)
@click.option('--batch_size', type=int)
@click.option('--model_url')
@click.option('--graph_url')
@click.option('--dev')
def train(num_hid, bidir, net_type, n2p, batch_size, model_url, graph_url, dev):
    '''
    Main function of the model
    '''

    global train_batch_size, MODEL_URL, GRAPH_URL, device

    if batch_size:
        train_batch_size = batch_size
        print("Setting up the batch size to {}.........................".format(batch_size))
    if model_url:
        MODEL_URL = model_url
        print("Setting up the model url to {}.........................".format(MODEL_URL))
    if graph_url:
        GRAPH_URL = graph_url
        print("Setting up the graph url to {}.........................".format(GRAPH_URL))

    if dev:
        device = dev
        print("Setting up the device to {}.........................".format(device))

    # 1 Load and process the input data including the ontology
    # Load the word embeddings
    word_vectors = load_word_vectors(WORD_VECTORS_URL)

    # Load the ontology and extract the feature vectors
    ontology, ontology_vectors, slots = load_ontology(ONTOLOGY_URL, word_vectors)

    # Load and process the training data
    dialogues, _ = load_woz_data(TRAINING_URL, word_vectors, ontology)
    no_dialogues = len(dialogues)

    # Load and process the validation data
    val_dialogues, _ = load_woz_data(VALIDATION_URL, word_vectors, ontology)

    # Generate the validation batch data
    val_data = generate_batch(val_dialogues, 0, len(val_dialogues), len(ontology))
    val_iterations = int(len(val_dialogues)/train_batch_size)

    # 2 Initialise and set up the model graph
    # Initialise the model
    graph = tf.Graph()
    with graph.as_default():
        model_variables = model_definition(ontology_vectors, len(ontology), slots, num_hidden=num_hid, bidir=bidir,
                                           net_type=net_type, dev=device)
        (user, sys_res, no_turns, user_uttr_len, sys_uttr_len, labels, domain_labels, domain_accuracy,
         slot_accuracy, value_accuracy, value_f1, train_step, keep_prob, _, _, _) = model_variables
        [precision, recall, value_f1] = value_f1
        saver = tf.train.Saver()
        if device == 'gpu':
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
        else:
            config = tf.ConfigProto(device_count={'GPU': 0})

        sess = tf.Session(config=config)
        if os.path.exists(MODEL_URL + ".index"):
            saver.restore(sess, MODEL_URL)
            print("Loading from an existing model {} ....................".format(MODEL_URL))
        else:
            if not os.path.exists("models"):
                os.makedirs("models")
                os.makedirs("graphs")
            init = tf.global_variables_initializer()
            sess.run(init)
            print("Create new model parameters.....................................")
        merged = tf.summary.merge_all()
        val_accuracy = tf.summary.scalar('validation_accuracy', value_accuracy)
        val_f1 = tf.summary.scalar('validation_f1_score', value_f1)
        train_writer = tf.summary.FileWriter(GRAPH_URL, graph)
        train_writer.flush()

    # 3 Perform an epoch of training
    last_update = -1
    best_f_score = -1
    for epoch in range(no_epochs):

        batch_size = train_batch_size
        sys.stdout.flush()
        iterations = math.ceil(no_dialogues/train_batch_size)
        start_time = time.time()
        val_i = 0
        shuffle(dialogues)
        for batch_id in range(iterations):

            if batch_id == iterations - 1 and no_dialogues % iterations != 0:
                batch_size = no_dialogues % train_batch_size

            batch_user, batch_sys, batch_labels, batch_domain_labels, batch_user_uttr_len, batch_sys_uttr_len,\
                batch_no_turns = generate_batch(dialogues, batch_id, batch_size, len(ontology))

            [_, summary, da, sa, va, vf, pr, re] = sess.run([train_step, merged, domain_accuracy, slot_accuracy,
                                                             value_accuracy, value_f1, precision, recall],
                                                            feed_dict={user: batch_user, sys_res: batch_sys,
                                                                       labels: batch_labels,
                                                                       domain_labels: batch_domain_labels,
                                                                       user_uttr_len: batch_user_uttr_len,
                                                                       sys_uttr_len: batch_sys_uttr_len,
                                                                       no_turns: batch_no_turns,
                                                                       keep_prob: 0.5})

            print("The accuracies for domain is {:.2f}, slot {:.2f}, value {:.2f}, f1_score {:.2f} precision {:.2f}"
                  " recall {:.2f} for batch {}".format(da, sa, va, vf, pr, re, batch_id + iterations * epoch))

            train_writer.add_summary(summary, start_batch + batch_id + iterations * epoch)

        # ================================ VALIDATION ==============================================

            if batch_id % batches_per_eval == 0 or batch_id == 0:
                if batch_id == 0:
                    print("Batch", "0", "to", batch_id, "took", round(time.time() - start_time, 2), "seconds.")

                else:
                    print("Batch", batch_id + iterations * epoch - batches_per_eval, "to",
                          batch_id + iterations * epoch, "took",
                          round(time.time() - start_time, 3), "seconds.")
                    start_time = time.time()

                _, _, v_acc, f1_score, sm1, sm2 = evaluate_model(sess, model_variables, val_data,
                                                                 [val_accuracy, val_f1], batch_id, val_i)
                val_i += 1
                val_i %= val_iterations
                train_writer.add_summary(sm1, start_batch + batch_id + iterations * epoch)
                train_writer.add_summary(sm2, start_batch + batch_id + iterations * epoch)
                stime = time.time()
                current_metric = f1_score
                print(" Validation metric:", round(current_metric, 5), " eval took",
                      round(time.time() - stime, 2), "last update at:", last_update, "/", iterations)

                # and if we got a new high score for validation f-score, we need to save the parameters:
                if current_metric > best_f_score:

                    last_update = batch_id + iterations * epoch + 1
                    print("\n ====================== New best validation metric:", round(current_metric, 4),
                          " - saving these parameters. Batch is:", last_update, "/", iterations,
                          "---------------- ===========  \n")

                    best_f_score = current_metric

                    saver.save(sess, MODEL_URL)

        print("The best parameters achieved a validation metric of", round(best_f_score, 4))


@cli.command()
@click.option('--num_hid', type=int)
@click.option('--bidir/--no-bidir', default=True)
@click.option('--net_type')
@click.option('--n2p', type=int)
@click.option('--batch_size', type=int)
@click.option('--model_url')
@click.option('--graph_url')
def test(num_hid, bidir, net_type, n2p, batch_size, model_url, graph_url):

    if not os.path.exists("results"):
        os.makedirs("results")

    global train_batch_size, MODEL_URL, GRAPH_URL
    if batch_size:
        train_batch_size = batch_size
        print("Setting up the batch size to {}.........................".format(batch_size))
    if model_url:
        MODEL_URL = model_url
        print("Setting up the model url to {}.........................".format(MODEL_URL))
    if graph_url:
        GRAPH_URL = graph_url
        print("Setting up the graph url to {}.........................".format(GRAPH_URL))

    # 1 Load and process the input data including the ontology
    # Load the word embeddings
    word_vectors = load_word_vectors(WORD_VECTORS_URL)

    # Load the ontology and extract the feature vectors
    ontology, ontology_vectors, slots = load_ontology(ONTOLOGY_URL, word_vectors)

    # Load and process the training data
    dialogues, actual_dialogues = load_woz_data(TESTING_URL, word_vectors, ontology)
    no_dialogues = len(dialogues)

    # 2 Initialise and set up the model graph
    # Initialise the model
    graph = tf.Graph()
    with graph.as_default():
        model_variables = model_definition(ontology_vectors, len(ontology), slots, num_hidden=num_hid, bidir=bidir,
                                           net_type=net_type, test=True, dev='cpu')
        (user, sys_res, no_turns, user_uttr_len, sys_uttr_len, labels, domain_labels, domain_accuracy,
         slot_accuracy, value_accuracy, value_f1, train_step, keep_prob, predictions,
         true_predictions, [y, _]) = model_variables
        [precision, recall, value_f1] = value_f1
        saver = tf.train.Saver()
        config = tf.ConfigProto(device_count={'GPU': 0})
        sess = tf.Session(config=config)
        saver.restore(sess, MODEL_URL)
        print("Loading from an existing model {} ....................".format(MODEL_URL))

    iterations = math.ceil(no_dialogues / train_batch_size)
    batch_size = train_batch_size
    [slot_acc, tot_accuracy] = [np.zeros(len(ontology), dtype="float32"), 0]
    slot_accurac = 0
    #value_accurac = np.zeros((len(slots),), dtype="float32")
    value_accurac = 0
    joint_accuracy = 0
    f1_score = 0
    preci = 0
    recal = 0
    processed_dialogues = []
    np.set_printoptions(threshold=np.nan)
    for batch_id in range(int(iterations)):

        if batch_id == iterations - 1:
            batch_size = no_dialogues - batch_id*train_batch_size

        batch_user, batch_sys, batch_labels, batch_domain_labels, batch_user_uttr_len, batch_sys_uttr_len, \
            batch_no_turns = generate_batch(dialogues, batch_id, batch_size, len(ontology))

        [da, sa, va, vf, pr, re, pred, true_pred, y_pred] = sess.run([domain_accuracy, slot_accuracy, value_accuracy,
                                                                      value_f1, precision, recall, predictions,
                                                                      true_predictions, y],
                                                                     feed_dict={user: batch_user, sys_res: batch_sys,
                                                                                labels: batch_labels,
                                                                                domain_labels: batch_domain_labels,
                                                                                user_uttr_len: batch_user_uttr_len,
                                                                                sys_uttr_len: batch_sys_uttr_len,
                                                                                no_turns: batch_no_turns,
                                                                                keep_prob: 1.0})

        true = sum([1 if np.array_equal(pred[k, :], true_pred[k, :]) and sum(true_pred[k, :]) > 0 else 0
                    for k in range(true_pred.shape[0])])
        actual = sum([1 if sum(true_pred[k, :]) > 0 else 0 for k in range(true_pred.shape[0])])
        ja = true/actual
        tot_accuracy += da
        # joint_accuracy += ja
        slot_accurac += sa
        if math.isnan(pr):
            pr = 0
        preci += pr
        recal += re
        if math.isnan(vf):
            vf = 0
        f1_score += vf
        # value_accurac += va
        slot_acc += np.mean(np.asarray(np.equal(pred, true_pred), dtype="float32"), axis=0)

        dialgs, va1, ja = track_dialogue(actual_dialogues[batch_id*train_batch_size:
                                                         batch_id*train_batch_size + batch_size], ontology, pred,y_pred)
        processed_dialogues += dialgs
        joint_accuracy += ja
        value_accurac += va1

        print("The accuracies for domain is {:.2f}, slot {:.2f}, value {:.2f}, other value {:.2f}, f1_score {:.2f} precision {:.2f}"
              " recall {:.2f}  for batch {}".format(da, sa, np.mean(va), va1, vf, pr, re, batch_id))

    print("End of evaluating the test set...........................................................................")

    slot_acc /= iterations
    # print("The accuracies for each slot:")
    # print(value_accurac/iterations)
    print("The overall accuracies for domain is"
          " {}, slot {}, value {}, f1_score {}, precision {},"
          " recall {}, joint accuracy {}".format(tot_accuracy/iterations, slot_accurac/iterations,
                                                 value_accurac/iterations, f1_score/iterations,
                                                 preci/iterations, recal/iterations, joint_accuracy/iterations))

    with open(RESULTS_URL, 'w') as f:
        json.dump(processed_dialogues, f, indent=4)


if __name__ == '__main__':
    cli()

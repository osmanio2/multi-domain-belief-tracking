
import tensorflow as tf
from tensorflow.python.client import device_lib

network = "lstm"
bidirect = True
lstm_num_hidden = 50
max_utterance_length = 40
vector_dimension = 300
max_no_turns = 22


def get_available_devs():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


class GRU(tf.nn.rnn_cell.RNNCell):
    '''
    Create a Gated Recurrent unit to unroll the network through time
    for combining the current and previous belief states
    '''
    def __init__(self, W_h, U_h, M_h, W_m, U_m, label_size, reuse=None, binary_output=False):
        super(GRU, self).__init__(_reuse=reuse)
        self.label_size = label_size
        self.M_h = M_h
        self.W_m = W_m
        self.U_m = U_m
        self.U_h = U_h
        self.W_h = W_h
        self.binary_output = binary_output

    def __call__(self, inputs, state, scope=None):
        state_only = tf.slice(state, [0, self.label_size], [-1, -1])
        output_only = tf.slice(state, [0, 0], [-1, self.label_size])
        new_state = tf.tanh(tf.matmul(inputs, self.U_m) + tf.matmul(state_only, self.W_m))
        output = tf.matmul(inputs, self.U_h) + tf.matmul(output_only, self.W_h) + tf.matmul(state_only, self.M_h)
        if self.binary_output:
            output_ = tf.sigmoid(output)
        else:
            output_ = tf.nn.softmax(output)
        state = tf.concat([output_, new_state], 1)
        return output, state


    @property
    def state_size(self):
        return tf.shape(self.W_m)[0] + self.label_size

    @property
    def output_size(self):
        return tf.shape(self.W_h)[0]


def define_CNN_model(utter, num_filters=300, name="r"):
    """
    Better code for defining the CNN model.
    """
    filter_sizes = [1, 2, 3]
    W = []
    b = []
    for i, filter_size in enumerate(filter_sizes):
        filter_shape = [filter_size, vector_dimension, 1, num_filters]
        W.append(tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="F_W"))
        b.append(tf.Variable(tf.constant(0.1, shape=[num_filters]), name="F_b"))

    utter = tf.reshape(utter, [-1, max_utterance_length, vector_dimension])

    hidden_representation = tf.zeros([num_filters], tf.float32)

    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        # with tf.name_scope("conv-maxpool-%s" % filter_size):
        # Convolution Layer
        conv = tf.nn.conv2d(
            tf.expand_dims(utter, -1),
            W[i],
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv_R")
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b[i]), name="relu")
        # Maxpooling over the outputs
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, max_utterance_length - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="r_")
        pooled_outputs.append(pooled)

        hidden_representation += tf.reshape(tf.concat(pooled, 3), [-1, num_filters])

    hidden_representation = tf.reshape(hidden_representation, [-1, max_no_turns, num_filters], name=name)

    return hidden_representation


def lstm_model(text_input, utterance_length, num_hidden, name, net_type, bidir):
    '''
    Define an Lstm model that will run across the user input and system act
    :param text_input: [batch_size, max_num_turns, max_utterance_size, vector_dimension]
    :param utterance_length: number words in every utterance [batch_size, max_num_turns, 1]
    :param num_hidden: -- int --
    :param name: The name of lstm network
    :param net_type: type of the network ("lstm" or "gru" or "rnn")
    :param bidir: use a bidirectional network -- bool --
    :return: output at each state [batch_size, max_num_turns, max_utterance_size, num_hidden],
     output of the final state [batch_size, max_num_turns, num_hidden]
    '''
    with tf.variable_scope(name):

        text_input = tf.reshape(text_input, [-1, max_utterance_length, vector_dimension])
        utterance_length = tf.reshape(utterance_length, [-1])

        def rnn(net_typ, num_units):
            if net_typ == "lstm":
                return tf.nn.rnn_cell.LSTMCell(num_units)
            elif net_typ == "gru":
                return tf.nn.rnn_cell.GRUCell(num_units)
            else:
                return tf.nn.rnn_cell.BasicRNNCell(num_units)

        if bidir:
            assert num_hidden % 2 == 0
            rev_cell = rnn(net_type, num_hidden//2)
            cell = rnn(net_type, num_hidden//2)
            _, lspd = tf.nn.bidirectional_dynamic_rnn(cell, rev_cell, text_input, dtype=tf.float32,
                                                      sequence_length=utterance_length)
            if net_type == "lstm":
                lspd = (lspd[0].h, lspd[1].h)

            last_state = tf.concat(lspd, 1)
        else:
            cell = rnn(net_type, num_hidden)
            _, last_state = tf.nn.dynamic_rnn(cell, text_input, dtype=tf.float32, sequence_length=utterance_length)
            if net_type == "lstm":
                last_state = last_state.h

        last_state = tf.reshape(last_state, [-1, max_no_turns, num_hidden])

        return last_state


def model_definition(ontology, num_slots, slots, num_hidden=None, net_type=None, bidir=None, test=False, dev=None):
    '''
    Create neural belief tracker model that is defined in my notes. It consists of encoding the user and system input,
    then use the ontology to decode the encoder in manner that detects if a domain-slot-value class is mentioned
    :param ontology: numpy array of the embedded vectors of the ontology [num_slots, 3*vector_dimension]
    :param num_slots: number of ontology classes --int--
    :param slots: indices of the values of each slot list of lists of ints
    :param num_hidden: Number of hidden units or dimension of the hidden space
    :param net_type: The type of the encoder network cnn, lstm, gru, rnn ...etc
    :param bidir: For recurrent networks should it be bidirectional
    :param test: This is testing mode (no back-propagation)
    :param dev: Device to run the model on (cpu or gpu)
    :return: All input variable/placeholders output metrics (precision, recall, f1-score) and trainer
    '''

    global lstm_num_hidden

    if not net_type:
        net_type = network
    else:
        print("Setting up the type of the network to {}..............................".format(net_type))
    if bidir == None:
        bidir = bidirect
    else:
        print("Setting up type of the recurrent network to bidirectional {}...........................".format(bidir))
    if num_hidden:
        lstm_num_hidden = num_hidden
        print("Setting up type of the dimension of the hidden space to {}.........................".format(num_hidden))

    ontology = tf.constant(ontology, dtype=tf.float32)

    # ----------------------------------- Define the input variables --------------------------------------------------
    user_input = tf.placeholder(tf.float32, [None, max_no_turns, max_utterance_length, vector_dimension], name="user")
    system_input = tf.placeholder(tf.float32, [None, max_no_turns, max_utterance_length, vector_dimension], name="sys")
    num_turns = tf.placeholder(tf.int32, [None], name="num_turns")
    user_utterance_lengths = tf.placeholder(tf.int32, [None, max_no_turns], name="user_sen_len")
    sys_utterance_lengths = tf.placeholder(tf.int32, [None, max_no_turns], name="sys_sen_len")
    labels = tf.placeholder(tf.float32, [None, max_no_turns, num_slots], name="labels")
    domain_labels = tf.placeholder(tf.float32, [None, max_no_turns, num_slots], name="domain_labels")
    # dropout placeholder, 0.5 for training, 1.0 for validation/testing:
    keep_prob = tf.placeholder("float")
    
    # ------------------------------------ Create the Encoder networks ------------------------------------------------
    devs = ['/device:CPU:0']
    if dev == 'gpu':
        devs = get_available_devs()

    if net_type == "cnn":
        with tf.device(devs[1%len(devs)]):
            # Encode the domain of the user input using a LSTM network
            usr_dom_en = define_CNN_model(user_input, num_filters=lstm_num_hidden, name="h_u_d")
            # Encode the domain of the system act using a LSTM network
            sys_dom_en = define_CNN_model(system_input, num_filters=lstm_num_hidden, name="h_s_d")
            
        with tf.device(devs[2%len(devs)]):
            # Encode the slot of the user input using a CNN network
            usr_slot_en = define_CNN_model(user_input, num_filters=lstm_num_hidden, name="h_u_s")
            # Encode the slot of the system act using a CNN network
            sys_slot_en = define_CNN_model(system_input, num_filters=lstm_num_hidden, name="h_s_s")
            # Encode the value of the user input using a CNN network
            usr_val_en = define_CNN_model(user_input, num_filters=lstm_num_hidden, name="h_u_v")
            # Encode the value of the system act using a CNN network
            sys_val_en = define_CNN_model(system_input, num_filters=lstm_num_hidden, name="h_s_v")
            # Encode the user using a CNN network
            usr_en = define_CNN_model(user_input, num_filters=lstm_num_hidden//5, name="h_u")

    else:

        with tf.device(devs[1%len(devs)]):
            # Encode the domain of the user input using a LSTM network
            usr_dom_en = lstm_model(user_input, user_utterance_lengths, lstm_num_hidden, "h_u_d", net_type, bidir)
            usr_dom_en = tf.nn.dropout(usr_dom_en, keep_prob, name="h_u_d_out")
            # Encode the domain of the system act using a LSTM network
            sys_dom_en = lstm_model(system_input, sys_utterance_lengths, lstm_num_hidden, "h_s_d", net_type, bidir)
            sys_dom_en = tf.nn.dropout(sys_dom_en, keep_prob, name="h_s_d_out")

        with tf.device(devs[2%len(devs)]):
            # Encode the slot of the user input using a LSTM network
            usr_slot_en = lstm_model(user_input, user_utterance_lengths, lstm_num_hidden, "h_u_s", net_type, bidir)
            usr_slot_en = tf.nn.dropout(usr_slot_en, keep_prob, name="h_u_s_out")
            # Encode the slot of the system act using a LSTM network
            sys_slot_en = lstm_model(system_input, sys_utterance_lengths, lstm_num_hidden, "h_s_s", net_type, bidir)
            sys_slot_en = tf.nn.dropout(sys_slot_en, keep_prob, name="h_s_s_out")
            # Encode the value of the user input using a LSTM network
            usr_val_en = lstm_model(user_input, user_utterance_lengths, lstm_num_hidden, "h_u_v", net_type, bidir)
            usr_val_en = tf.nn.dropout(usr_val_en, keep_prob, name="h_u_v_out")
            # Encode the value of the system act using a LSTM network
            sys_val_en = lstm_model(system_input, sys_utterance_lengths, lstm_num_hidden, "h_s_v", net_type, bidir)
            sys_val_en = tf.nn.dropout(sys_val_en, keep_prob, name="h_s_v_out")
            # Encode the user using a LSTM network
            usr_en = lstm_model(user_input, user_utterance_lengths, lstm_num_hidden//5, "h_u", net_type, bidir)
            usr_en = tf.nn.dropout(usr_en, keep_prob, name="h_u_out")

    with tf.device(devs[1%len(devs)]):
        usr_dom_en = tf.tile(tf.expand_dims(usr_dom_en, axis=2), [1, 1, num_slots, 1], name="h_u_d")
        sys_dom_en = tf.tile(tf.expand_dims(sys_dom_en, axis=2), [1, 1, num_slots, 1], name="h_s_d")
    with tf.device(devs[2%len(devs)]):
        usr_slot_en = tf.tile(tf.expand_dims(usr_slot_en, axis=2), [1, 1, num_slots, 1], name="h_u_s")
        sys_slot_en = tf.tile(tf.expand_dims(sys_slot_en, axis=2), [1, 1, num_slots, 1], name="h_s_s")
        usr_val_en = tf.tile(tf.expand_dims(usr_val_en, axis=2), [1, 1, num_slots, 1], name="h_u_v")
        sys_val_en = tf.tile(tf.expand_dims(sys_val_en, axis=2), [1, 1, num_slots, 1], name="h_s_v")
        usr_en = tf.tile(tf.expand_dims(usr_en, axis=2), [1, 1, num_slots, 1], name="h_u")

    # All encoding vectors have size [batch_size, max_turns, num_slots, num_hidden]

    # Matrix that transforms the ontology from the embedding space to the hidden representation
    with tf.device(devs[1%len(devs)]):
        W_onto_domain = tf.Variable(tf.random_normal([vector_dimension, lstm_num_hidden]), name="W_onto_domain")
        W_onto_slot = tf.Variable(tf.random_normal([vector_dimension, lstm_num_hidden]), name="W_onto_slot")
        W_onto_value = tf.Variable(tf.random_normal([vector_dimension, lstm_num_hidden]), name="W_onto_value")

        # And biases
        b_onto_domain = tf.Variable(tf.zeros([lstm_num_hidden]), name="b_onto_domain")
        b_onto_slot = tf.Variable(tf.zeros([lstm_num_hidden]), name="b_onto_slot")
        b_onto_value = tf.Variable(tf.zeros([lstm_num_hidden]), name="b_onto_value")

        # Apply the transformation from the embedding space of the ontology to the hidden space
        domain_vec = tf.slice(ontology, begin=[0, 0], size=[-1, vector_dimension])
        slot_vec = tf.slice(ontology, begin=[0, vector_dimension], size=[-1, vector_dimension])
        value_vec = tf.slice(ontology, begin=[0, 2*vector_dimension], size=[-1, vector_dimension])
        # Each [num_slots, vector_dimension]
        d = tf.nn.dropout(tf.tanh(tf.matmul(domain_vec, W_onto_domain) + b_onto_domain), keep_prob, name="d")
        s = tf.nn.dropout(tf.tanh(tf.matmul(slot_vec, W_onto_slot) + b_onto_slot), keep_prob, name="s")
        v = tf.nn.dropout(tf.tanh(tf.matmul(value_vec, W_onto_value) + b_onto_value), keep_prob, name="v")
        # Each [num_slots, num_hidden]

        # Apply the comparison mechanism for all the user and system utterances and ontology values
        domain_user = tf.multiply(usr_dom_en, d, name="domain_user")
        domain_sys = tf.multiply(sys_dom_en, d, name="domain_sys")
        slot_user = tf.multiply(usr_slot_en, s, name="slot_user")
        slot_sys = tf.multiply(sys_slot_en, s, name="slot_sys")
        value_user = tf.multiply(usr_val_en, v, name="value_user")
        value_sys = tf.multiply(sys_val_en, v, name="value_sys")
        # All of size [batch_size, max_turns, num_slots, num_hidden]

        # -------------- Domain Detection -------------------------------------------------------------------------
        W_domain = tf.Variable(tf.random_normal([2*lstm_num_hidden]), name="W_domain")
        b_domain = tf.Variable(tf.zeros([1]), name="b_domain")
        y_d = tf.sigmoid(tf.reduce_sum(tf.multiply(tf.concat([domain_user, domain_sys], axis=3), W_domain), axis=3)
                         + b_domain) # [batch_size, max_turns, num_slots]

    # -------- Run through each of the 3 case ( inform, request, confirm) and decode the inferred state ---------
    # 1 Inform (User is informing the system about the goal, e.g. "I am looking for a place to stay in the centre")
    W_inform = tf.Variable(tf.random_normal([2 * lstm_num_hidden]), name="W_inform")
    b_inform = tf.Variable(tf.random_normal([1]), name="b_inform")
    inform = tf.add(tf.reduce_sum(tf.multiply(tf.concat([slot_user, value_user], axis=3), W_inform), axis=3), b_inform,
                    name="inform")  # [batch_size, max_turns, num_slots]

    # 2 Request (The system is requesting information from the user, e.g. "what type of food would you like?")
    with tf.device(devs[2%len(devs)]):
        W_request = tf.Variable(tf.random_normal([2 * lstm_num_hidden]), name="W_request")
        b_request = tf.Variable(tf.random_normal([1]), name="b_request")
        request = tf.add(tf.reduce_sum(tf.multiply(tf.concat([slot_sys, value_user], axis=3), W_request), axis=3),
                         b_request, name="request")  # [batch_size, max_turns, num_slots]

    # 3 Confirm (The system is confirming values given by the user, e.g. "How about turkish food?")
    with tf.device(devs[3%len(devs)]):
        size = 2 * lstm_num_hidden + lstm_num_hidden//5
        W_confirm = tf.Variable(tf.random_normal([size]), name="W_confirm")
        b_confirm = tf.Variable(tf.random_normal([1]), name="b_confirm")
        confirm = tf.add(tf.reduce_sum(tf.multiply(tf.concat([slot_sys, value_sys, usr_en], axis=3), W_confirm), axis=3),
                         b_confirm, name="confirm")  # [batch_size, max_turns, num_slots]

    output = inform + request + confirm

    # -------------------- Adding the belief update RNN with memory cell (Taken from previous model) -------------------
    with tf.device(devs[2%len(devs)]):
        domain_memory = tf.Variable(tf.random_normal([1, 1]), name="domain_memory")
        domain_current = tf.Variable(tf.random_normal([1, 1]), name="domain_current")
        domain_M_h = tf.Variable(tf.random_normal([1, 1]), name="domain_M_h")
        domain_W_m = tf.Variable(tf.random_normal([1, 1], name="domain_W_m"))
        domain_U_m = tf.Variable(tf.random_normal([1, 1]), name="domain_U_m")
    a_memory = tf.Variable(tf.random_normal([1, 1]), name="a_memory")
    b_memory = tf.Variable(tf.random_normal([1, 1]), name="b_memory")
    a_current = tf.Variable(tf.random_normal([1, 1]), name="a_current")
    b_current = tf.Variable(tf.random_normal([1, 1]), name="b_current")
    M_h_a = tf.Variable(tf.random_normal([1, 1]), name="M_h_a")
    M_h_b = tf.Variable(tf.random_normal([1, 1]), name="M_h_b")
    W_m_a = tf.Variable(tf.random_normal([1, 1]), name="W_m_a")
    W_m_b = tf.Variable(tf.random_normal([1, 1]), name="W_m_b")
    U_m_a = tf.Variable(tf.random_normal([1, 1]), name="U_m_a")
    U_m_b = tf.Variable(tf.random_normal([1, 1]), name="U_m_b")

    # ---------------------------------- Unroll the domain over time --------------------------------------------------
    with tf.device(devs[1%len(devs)]):
        cell = GRU(domain_memory*tf.diag(tf.ones(num_slots)), domain_current*tf.diag(tf.ones(num_slots)),
                   domain_M_h*tf.diag(tf.ones(num_slots)), domain_W_m*tf.diag(tf.ones(num_slots)),
                   domain_U_m*tf.diag(tf.ones(num_slots)), num_slots,
                   binary_output=True)

        y_d, _ = tf.nn.dynamic_rnn(cell, y_d, sequence_length=num_turns, dtype=tf.float32)

        domain_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=domain_labels, logits=y_d), axis=2,
                                    name="domain_loss") / (num_slots/len(slots))

        y_d = tf.sigmoid(y_d)

    with tf.device(devs[0%len(devs)]):

        loss = [None for _ in range(len(slots))]
        slot_pred = [None for _ in range(len(slots))]
        slot_label = [None for _ in range(len(slots))]
        val_pred = [None for _ in range(len(slots))]
        val_label = [None for _ in range(len(slots))]
        y = [None for _ in range(len(slots))]
        y_pred = [None for _ in range(len(slots))]
        for i in range(len(slots)):

            num_values = slots[i] + 1  # For the none case
            size = sum(slots[:i+1])-slots[i]
            if test:
                domain_output = tf.slice(tf.round(y_d), begin=[0, 0, size], size=[-1, -1, slots[i]])
            else:
                domain_output = tf.slice(domain_labels, begin=[0, 0, size], size=[-1, -1, slots[i]])
            max_val = tf.expand_dims(tf.reduce_max(domain_output, axis=2), axis=2)
            tf.assert_less_equal(max_val, 1.0)
            tf.assert_equal(tf.round(max_val), max_val)
            domain_output = tf.concat([tf.zeros(tf.shape(domain_output)), 1 - max_val], axis=2)

            slot_output = tf.slice(output, begin=[0, 0, size], size=[-1, -1, slots[i]])
            slot_output = tf.concat([slot_output, tf.zeros([tf.shape(output)[0], max_no_turns, 1])], axis=2)

            labels_output = tf.slice(labels, begin=[0, 0, size], size=[-1, -1, slots[i]])
            max_val = tf.expand_dims(tf.reduce_max(labels_output, axis=2), axis=2)
            tf.assert_less_equal(max_val, 1.0)
            tf.assert_equal(tf.round(max_val), max_val)
            slot_label[i] = max_val
            # [Batch_size, max_turns, 1]
            labels_output = tf.argmax(tf.concat([labels_output, 1 - max_val], axis=2), axis=2)
            # [Batch_size, max_turns]
            val_label[i] = tf.cast(tf.expand_dims(labels_output, axis=2), dtype="float")
            # [Batch_size, max_turns, 1]

            diag_memory = a_memory * tf.diag(tf.ones(num_values))
            non_diag_memory = tf.matrix_set_diag(b_memory * tf.ones([num_values, num_values]), tf.zeros(num_values))
            W_memory = diag_memory + non_diag_memory

            diag_current = a_current * tf.diag(tf.ones(num_values))
            non_diag_current = tf.matrix_set_diag(b_current * tf.ones([num_values, num_values]), tf.zeros(num_values))
            W_current = diag_current + non_diag_current

            diag_M_h = M_h_a * tf.diag(tf.ones(num_values))
            non_diag_M_h = tf.matrix_set_diag(M_h_b * tf.ones([num_values, num_values]), tf.zeros(num_values))
            M_h = diag_M_h + non_diag_M_h

            diag_U_m = U_m_a * tf.diag(tf.ones(num_values))
            non_diag_U_m = tf.matrix_set_diag(U_m_b * tf.ones([num_values, num_values]), tf.zeros(num_values))
            U_m = diag_U_m + non_diag_U_m

            diag_W_m = W_m_a * tf.diag(tf.ones(num_values))
            non_diag_W_m = tf.matrix_set_diag(W_m_b * tf.ones([num_values, num_values]), tf.zeros(num_values))
            W_m = diag_W_m + non_diag_W_m

            cell = GRU(W_memory, W_current, M_h, W_m, U_m, num_values)
            y_predict, _ = tf.nn.dynamic_rnn(cell, slot_output, sequence_length=num_turns, dtype=tf.float32)

            y_predict = y_predict + 1000000.0*domain_output
            # [Batch_size, max_turns, num_values]

            y[i] = tf.nn.softmax(y_predict)
            val_pred[i] = tf.cast(tf.expand_dims(tf.argmax(y[i], axis=2), axis=2), dtype="float32")
            # [Batch_size, max_turns, 1]
            y_pred[i] = tf.slice(tf.one_hot(tf.argmax(y[i], axis=2), dtype=tf.float32, depth=num_values),
                                 begin=[0, 0, 0], size=[-1, -1, num_values - 1])
            y[i] = tf.slice(y[i], begin=[0, 0, 0], size=[-1, -1, num_values - 1])
            slot_pred[i] = tf.cast(tf.reduce_max(y_pred[i], axis=2, keep_dims=True), dtype="float32")
            # [Batch_size, max_turns, 1]
            loss[i] = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_output, logits=y_predict)
            # [Batch_size, max_turns]

    # ---------------- Compute the output and the loss function (cross_entropy) and add to optimizer--------------------
    cross_entropy = tf.add_n(loss, name="cross_entropy")
    # Add the error from the domains
    cross_entropy = tf.add(cross_entropy, domain_loss, name="total_loss")

    y = tf.concat(y, axis=2, name="y")

    mask = tf.cast(tf.sequence_mask(num_turns, maxlen=max_no_turns), dtype=tf.float32)
    mask_extended = tf.tile(tf.expand_dims(mask, axis=2), [1, 1, num_slots])
    cross_entropy = tf.reduce_sum(mask*cross_entropy, axis=1)/tf.cast(num_turns, dtype=tf.float32)

    optimizer = tf.train.AdamOptimizer(0.001)
    train_step = optimizer.minimize(cross_entropy, colocate_gradients_with_ops=True)

    # ----------------- Get the precision, recall f1-score and accuracy -----------------------------------------------

    # Domain accuracy
    true_predictions = tf.reshape(domain_labels, [-1, num_slots])
    predictions = tf.reshape(tf.round(y_d) * mask_extended, [-1, num_slots])

    y_d = tf.reshape(y_d * mask_extended, [-1, num_slots])

    _, _, _, domain_accuracy = get_metrics(predictions, true_predictions, num_turns, mask_extended, num_slots)

    mask_extended_2 = tf.tile(tf.expand_dims(mask, axis=2), [1, 1, len(slots)])

    # Slot accuracy
    true_predictions = tf.reshape(tf.concat(slot_label, axis=2), [-1, len(slots)])
    predictions = tf.reshape(tf.concat(slot_pred, axis=2) * mask_extended_2, [-1, len(slots)])

    _, _, _, slot_accuracy = get_metrics(predictions, true_predictions, num_turns, mask_extended_2, len(slots))

    # accuracy
    if test:
        value_accuracy = []
        mask_extended_3 = tf.expand_dims(mask, axis=2)
        for i in range(len(slots)):
            true_predictions = tf.reshape(val_label[i] * mask_extended_3, [-1, 1])
            predictions = tf.reshape(val_pred[i] * mask_extended_3, [-1, 1])

            _, _, _, value_acc = get_metrics(predictions, true_predictions, num_turns, mask_extended_3, 1)
            value_accuracy.append(value_acc)

        value_accuracy = tf.stack(value_accuracy)
    else:
        true_predictions = tf.reshape(tf.concat(val_label, axis=2) * mask_extended_2, [-1, len(slots)])
        predictions = tf.reshape(tf.concat(val_pred, axis=2) * mask_extended_2, [-1, len(slots)])

        _, _, _, value_accuracy = get_metrics(predictions, true_predictions, num_turns, mask_extended_2, len(slots))

    # Value f1score a
    true_predictions = tf.reshape(labels, [-1, num_slots])
    predictions = tf.reshape(tf.concat(y_pred, axis=2) * mask_extended, [-1, num_slots])

    precision, recall, value_f1_score, _ = get_metrics(predictions, true_predictions, num_turns,
                                                       mask_extended, num_slots)

    y_ = tf.reshape(y, [-1, num_slots])

    # -------------------- Summarise the statistics of training to be viewed in tensorboard-----------------------------
    tf.summary.scalar("domain_accuracy", domain_accuracy)
    tf.summary.scalar("slot_accuracy", slot_accuracy)
    tf.summary.scalar("value_accuracy", value_accuracy)
    tf.summary.scalar("value_f1_score", value_f1_score)
    tf.summary.scalar("cross_entropy", tf.reduce_mean(cross_entropy))

    value_f1_score = [precision, recall, value_f1_score]

    return user_input, system_input, num_turns, user_utterance_lengths, sys_utterance_lengths, labels, domain_labels,\
        domain_accuracy, slot_accuracy, value_accuracy, value_f1_score, train_step, keep_prob, predictions,\
        true_predictions, [y_, y_d]


def get_metrics(predictions, true_predictions, no_turns, mask, num_slots):
    mask = tf.reshape(mask, [-1, num_slots])
    correct_prediction = tf.cast(tf.equal(predictions, true_predictions), "float32") * mask

    num_positives = tf.reduce_sum(true_predictions)
    classified_positives = tf.reduce_sum(predictions)

    true_positives = tf.multiply(predictions, true_predictions)
    num_true_positives = tf.reduce_sum(true_positives)

    recall = num_true_positives / num_positives
    precision = num_true_positives / classified_positives
    f_score = (2 * recall * precision) / (recall + precision)
    accuracy = tf.reduce_sum(correct_prediction)/(tf.cast(tf.reduce_sum(no_turns), dtype="float32")*num_slots)

    return precision, recall, f_score, accuracy

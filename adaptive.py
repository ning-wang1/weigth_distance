import argparse
import tensorflow as tf

def weight_constrain(args, loss1, mal_loss1, agent_model, constrain_weights, t):
    """
    add regulation to the loss optimization. loss = loss + rho*|w-w_global|
    :param loss1:
    :param mal_loss1:
    :param agent_model:
    :param constrain_weights:
    :param t:
    :return:
    """
    # Adding weight based regularization
    loss2 = tf.constant(0.0)
    # mal_loss2 = tf.constant(0.0)
    layer_count = 0
    if 'dist_oth' in args.mal_strat and t < 1:
        rho = 0.0
    else:
        rho = args.rho
    for layer in agent_model.layers:
        counter = 0
        for weight in layer.weights:
            # print(counter)
            constrain_weight_curr = tf.convert_to_tensor(constrain_weights[layer_count], dtype=tf.float32)
            delta_constrain = (weight - constrain_weight_curr)
            if 'wt_o' in args.mal_strat:  # wt_o means weight only. Bias is not considered
                if counter % 2 == 0:
                    loss2 += tf.nn.l2_loss(delta_constrain)
            else:
                loss2 += tf.nn.l2_loss(delta_constrain)
            layer_count += 1
            counter += 1
    loss = loss1 + rho * loss2
    mal_loss = mal_loss1

    return loss, loss2, mal_loss

def train_model(args, loss):
    if 'adam' in args.optimizer:
        optimizer = tf.train.AdamOptimizer(learning_rate=args.eta).minimize(loss)
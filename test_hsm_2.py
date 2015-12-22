import theano
import theano.tensor as T
import random
import numpy
import hierarchical_softmax

def create_data(label_count, example_size):

    the_examples_and_labels = []

    for l in range(label_count):
        base_example = []
        for i in range(example_size):
            base_example.append(random.uniform(-1.0,1.0))
        for i in range(100):
            the_examples_and_labels.append((numpy.array(corrupt(base_example)), l))

    return the_examples_and_labels

def corrupt(vector):

    new_vector = []
    for v in vector:
        new_vector.append(v + random.uniform(-0.05, 0.05))
    return new_vector

def main():

    data = create_data(100, 50)
    random.shuffle(data)
    minibatches = []
    for i in range(0,len(data),100):
        examples = []
        labels = []
        for b in range(i, i+100):
            examples.append(data[b][0])
            labels.append(data[b][1])
        minibatches.append((numpy.array(examples), numpy.array(labels)))

    #Parallel hs-test
    tree = hierarchical_softmax.build_binary_tree(range(100))
    hs = hierarchical_softmax.HierarchicalSoftmax(tree,50,100)

    pf = hs.get_probability_function()

    print 'Minibatch probabilities before training'
    print T.exp(pf(minibatches[0][0], minibatches[0][1])[0]).eval()

    #Hierarchical softmax test
    tree = hierarchical_softmax.build_binary_tree(range(100))
    hs = hierarchical_softmax.HierarchicalSoftmax(tree,50,100)

    train_f = hs.get_training_function()

    hc = []
    for i in range(10):
        for mb in minibatches[1:]:
            hc.append(train_f(mb[0], mb[1]))
        print numpy.mean(hc), i

    pf = hs.get_probability_function()
    print 'Minibatch probabilities after training'
    print T.exp(pf(minibatches[0][0], minibatches[0][1])[0]).eval()

    print
    print 'Predictions'

    red = hs.label_tool(minibatches[0][0])
    for o, r in zip(minibatches[0][1],red[1][-1]):
        print o, r

    #Logistic regression with normal softmax-test
    input = T.dmatrix()
    y = T.lvector()
    rng = numpy.random.RandomState(1234)

    learning_rate = 0.5
    W_log = theano.shared(value=numpy.asarray(rng.uniform(low=-numpy.sqrt(6. / 50), high=numpy.sqrt(6. / 50),size=(50, 100)),dtype=theano.config.floatX), name = 'W_soft', borrow = True)
    b_log = theano.shared(value=numpy.zeros(100))
    p_y_given_x = T.nnet.softmax(T.dot(input, W_log) + b_log)
    cost = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])
    params = [W_log, b_log]
    gparams = [T.grad(cost, param) for param in params]
    updates = [(param, param - learning_rate * gparam) for param, gparam in zip(params, gparams)]
    train_f = theano.function(inputs=[input, y], outputs=[cost], updates=updates)

    print
    print 'Ordinary softmax'

    hc = []
    for i in range(10):
        for mb in minibatches[1:]:
            hc.append(train_f(mb[0], mb[1]))
        print numpy.mean(hc), i


main()

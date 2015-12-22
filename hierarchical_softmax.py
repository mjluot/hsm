import numpy
import theano, theano.tensor as T
import random
import time
import theano.sparse
import theano.sandbox

#Tree stuff

class TreeNode():

    def __init__(self, index=None, right=None, left=None, parent=None, parent_choice=None):

        self.index = index
        self.right = right
        self.left = left
        self.parent = parent
        self.parent_choice = parent_choice

    def __repr__(self):
        return '<' + str(self.index) + ', 0:' + str(self.left.index) + ', 1:' + str(self.right.index) + '>'

class ResultNode():

    def __init__(self, value=None, parent=None):

        self.value = value
        self.parent = parent
        self.index = 'res:' + str(self.value)

    def __repr__(self):
        return '<' + str(self.value) + '>'

def build_binary_tree(values):

    current_layer = []
    for v in values:
        current_layer.append(ResultNode(value=v))
    layers = [current_layer,]
    count = 0
    while(len(current_layer) > 1):
        pairs = []
	if len(current_layer) > 1:
		while(len(current_layer) > 1):
		    pairs.append(current_layer[:2])
		    current_layer = current_layer[2:]
	else:
		pairs = [current_layer]
                current_layer = []
        new_layer = []
        for p in pairs:
            tn = TreeNode(index=count, right=p[1], left=p[0])
            count += 1
            p[1].parent = tn
            p[1].parent_choice = 1
            p[0].parent = tn
            p[0].parent_choice = -1
            new_layer.append(tn)
        if len(current_layer) > 0:
            new_layer.extend(current_layer)
            current_layer = []
        layers.append(new_layer)
        current_layer = new_layer

    return layers

class HierarchicalSoftmax():

    def __init__(self, tree, size, mb_size):

        self.mb_size = mb_size
        self.learning_rate = 0.5
        self.size = size
        self.rng = numpy.random.RandomState(1234)
        #Make routes
        self.tree = tree

        self.nodes = []
        self.node_dict = {}
        self.result_dict = {}
        self.routes = []

        self.label_count = 0
        self.node_count = 0
        for layer in tree:
            for i in layer:
                if isinstance(i, TreeNode):
                    self.node_count += 1
                    self.nodes.append(i)
                    self.node_dict[i.index] = i

                if isinstance(i, ResultNode):
                    self.label_count += 1
                    self.result_dict[i.value] = i

        #Lets also put the tree into a matrix
        #

        tree_matrix_val = numpy.ones((self.node_count + self.label_count, 4), dtype=numpy.int) * -1
        for layer in tree[::-1]:
            for i in layer:
                if isinstance(i, TreeNode):
                    try:
                        if not isinstance(i.left.index, str):
                            tree_matrix_val[i.index][0] = i.left.index
                        else:
                            tree_matrix_val[i.index][0] = i.index#self.node_count + int(i.left.index.split(':')[-1]) + 1
                            tree_matrix_val[i.index][2] = int(i.left.index.split(':')[-1])

                        if not isinstance(i.right.index, str):
                            tree_matrix_val[i.index][1] = i.right.index
                        else:
                            tree_matrix_val[i.index][1] = i.index#self.node_count + int(i.right.index.split(':')[-1]) + 1
                            tree_matrix_val[i.index][3] = int(i.right.index.split(':')[-1])
                    except:
                        pass#import pdb;pdb.set_trace()

                #if isinstance(i, ResultNode):
                #    tree_matrix_val[i.value + self.node_count][0] = i.value + self.node_count
                #    tree_matrix_val[i.value + self.node_count][1] = i.value + self.node_count
                #    tree_matrix_val[i.value + self.node_count][2] = i.value

        self.max_route_len = 0
        for u in sorted(self.result_dict.keys()):
            self.routes.append(self.get_route(self.result_dict[u]))
            if len(self.routes[-1]) > self.max_route_len:
                self.max_route_len = len(self.routes[-1])


        self.route_node_matrix_val = numpy.zeros((len(self.result_dict.keys()), self.max_route_len), dtype=numpy.int)
        self.route_choice_matrix_val = numpy.zeros((len(self.result_dict.keys()), self.max_route_len, ), dtype=numpy.int)
        self.mask_matrix_val = numpy.zeros((len(self.result_dict.keys()), self.max_route_len, ), dtype=numpy.int)

        #import pdb; pdb.set_trace()
        #Routes matrix
        #Mask-matrix
        for i, route in enumerate(self.routes):
            for a in range(self.max_route_len):
                try:
                    self.route_node_matrix_val[i][a] = route[a][0].index
                    self.route_choice_matrix_val[i][a] = route[a][1]
                    self.mask_matrix_val[i][a] = 1.0
                except:
                    self.route_node_matrix_val[i][a] = 0
                    self.route_choice_matrix_val[i][a] = 0
                    self.mask_matrix_val[i][a] = 0.0

        self.tree_matrix = theano.shared(value=tree_matrix_val, name = 'tree_matrix', borrow = True)
        self.route_node_matrix = theano.shared(value=self.route_node_matrix_val, name = 'route_node_matrix', borrow = True)
        self.route_choice_matrix = theano.shared(value=self.route_choice_matrix_val, name = 'route_choice_matrix', borrow = True)
        self.mask_matrix = theano.shared(value=self.mask_matrix_val, name = 'route_mask_matrix', borrow = True)

        #Parameter_matrix_W
        #Make this a little nicer
        wp_val=numpy.asarray(self.rng.uniform(low=-numpy.sqrt(6. / (size + 2)), high=numpy.sqrt(6. / (size + 2)),size=(len(self.nodes) + 1, size)),dtype=theano.config.floatX)
        self.wp_matrix = theano.shared(value=wp_val,name='V_soft',borrow=True)
        #Parameter_matrix_b
        #self.bp_matrix = theano.shared(value=numpy.zeros((len(self.nodes), 2),dtype=theano.config.floatX),name='b_soft',borrow=True)

        #Let us build the graph

        self.y = T.lvector()
        self.x = T.dmatrix()

        n_node_route = self.route_node_matrix[self.y]
        n_choice_route = self.route_choice_matrix[self.y]
        n_mask = self.mask_matrix[self.y]

        #1.
        nodes = self.route_node_matrix[self.y]
        choices = self.route_choice_matrix[self.y]
        mask = self.mask_matrix[self.y]

        #2.
        wp = self.wp_matrix[nodes]

        #3. Let's make the gemv

        batch_size = self.x.shape[0]
        vec_size = self.x.shape[1]
        route_size = n_choice_route.shape[1]

        #output shape
        o = T.zeros((batch_size, route_size, 1))
        #let's 
        ewps = wp.reshape((batch_size, route_size, vec_size, 1))
        ewp = theano.function([self.x, self.y], ewps)

        #Check these
        idx = T.arange(batch_size).reshape((batch_size,1))
        ebin = []
        for i in range(self.mb_size):
            ebin.append(numpy.arange(self.max_route_len))
        odx = T.as_tensor_variable(numpy.asarray(ebin)) 
        iv = self.x.reshape((self.x.shape[0], 1, self.x.shape[1]))

        gb = theano.sandbox.blocksparse.SparseBlockGemv()
        node = gb.make_node(o, ewps, iv, idx, odx)
        matrix_f = theano.function([self.x, self.y], node.outputs[0])

        #The dots are done, now is time of direction and the mask
        dots_with_choice = node.outputs[0].reshape((batch_size ,route_size)) * choices
        log_sig = T.log(T.nnet.sigmoid(dots_with_choice)) * mask
        self.sums = T.sum(log_sig, axis=1)

        self.cost = -T.mean(self.sums)
        params = [self.wp_matrix,]

        gparams = [T.grad(self.cost, param) for param in params]
        updates = [(param, param - self.learning_rate * gparam) for param, gparam in zip(params, gparams)]
        self.train_f = theano.function(inputs=[self.x, self.y], outputs=[self.cost], updates=updates)

        #Okay and then the prediction function
        #So this, instead of having a route will have to iterate over

        self.input_vectors_m = T.dmatrix()
        self.root_node = T.as_tensor_variable(numpy.array([self.tree[-1][0].index]))
        self.node_count_t = T.as_tensor_variable(self.node_count)

        def istep():
            return self.root_node
        ires, _ = theano.scan(fn=istep, n_steps=self.x.shape[0])
        fires = ires.flatten()

        def predict_step(current_node, input_vector):

            #Get the results
            node_res_l = T.nnet.sigmoid(T.dot(self.wp_matrix[current_node], input_vector.T))
            correct_nodes_l = node_res_l[T.arange(input_vector.shape[0]),T.arange(input_vector.shape[0])]

            node_res_r = T.nnet.sigmoid(-1 * T.dot(self.wp_matrix[current_node], input_vector.T))
            correct_nodes_r = node_res_r[T.arange(input_vector.shape[0]),T.arange(input_vector.shape[0])]

            choice = correct_nodes_l > correct_nodes_r

            next_node = self.tree_matrix[current_node.flatten(), choice.flatten()]
            labelings = self.tree_matrix[current_node.flatten(), choice.flatten() + 2]

            return next_node, labelings, choice

        outputs_info = [fires, None, None]
        xresult, _ = theano.scan(fn=predict_step, outputs_info=outputs_info, sequences = [], non_sequences=self.x, n_steps=self.max_route_len)
        self.labels = xresult[1][-1][-1]
        self.predict_labels = theano.function([self.x],[self.labels])
        self.label_tool = theano.function([self.x], xresult)

    def get_prediction_functions(self):
        return self.predict_labels, self.label_tool

    def get_training_function(self):
        return self.train_f

    def get_x_and_y(self):
        return self.x, self.y

    def get_output(self):
        pass

    def get_probability_function(self):
        return theano.function([self.x, self.y], [self.sums])

    def get_route(self, i):
        route = []
        parent = i.parent
        parent_choice = i.parent_choice
        route.append((parent, parent_choice))
        while(parent != None):
            n_parent = parent.parent
            if n_parent != None:
                parent_choice = parent.parent_choice
                route.append((n_parent, parent_choice))
            parent = parent.parent #Hahaha :D
        return route


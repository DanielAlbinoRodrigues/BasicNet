import numpy as np
import math
import pickle as pkl

class Net:

    def __init__(self, layers, seed=None, print_output=False):

        # Seed RNG if desired
        if seed is not None:
            np.random.seed(seed)

        # Throw exception if the input layer has an activation associated with it
        if type(layers[0].activate).__name__ != 'NoActivation':
            raise ValueError('First layer must have no activation associated with it.')

        self.layers = layers
        self.n_inputs = layers[0].get_neuron_count()
        self.n_outputs = layers[len(layers)-1].get_neuron_count()
        self.train_hist = None
        self.print_output = print_output

        # Initialize weights
        self.weights = []
        self.reset_weights()

    # Get total number of layers in network
    def get_num_layers(self):
        return len(self.layers)

    # Resets weights for the model
    def reset_weights(self):

        for p in range(len(self.layers)-1):

            prev_count = self.layers[p].get_neuron_count()
            next_count = self.layers[p+1].get_neuron_count()

            r = self.layers[p+1].xavier_weight_distrib(prev_count,  next_count)

            # Add 1 to the prev_count to act as a bias
            self.weights.append(np.random.rand(next_count, prev_count+1)*2*r-r)

    # Returns all activations for all layers
    def full_predict(self, X):

        # Check input length
        if len(X) != self.n_inputs:
            raise ValueError('Input vector must be the same length as number of input neurons.')

        # Append a one to the incoming input vector as initial input bias
        X = np.append(X, 1)

        # Initialize list of layer outputs
        outputs = [X.tolist()]

        # Initialize list of unactivated layer outputs
        unactivated_outputs = [X.tolist()]

        # Loop through layers
        for k in range(1, self.get_num_layers()):

            # Initialize the outputs that will feed the next layer
            new_X = []
            unactivated_new_X = []

            # Loop through neurons in this layer k
            for j in range(self.layers[k].get_neuron_count()):  # Loop over each output node

                # Determine count of incoming signals per each neuron in this layer. Don't forget to add one for bias.
                input_count = self.layers[k-1].get_neuron_count() + 1

                # Sum inputs for this neuron
                neuron_sum = 0
                for i in range(input_count):

                    w = self.weights[k-1][j, i]
                    neuron_sum += w*X[i]

                # Apply activation function
                activated_neuron_sum = self.layers[k].activation(neuron_sum)

                # Append to output vector
                unactivated_new_X.append(neuron_sum)
                new_X.append(activated_neuron_sum)

            # Append a one for bias here, unless we are on the final output layer
            if(k!=self.get_num_layers()-1):
                new_X.append(1)
                unactivated_new_X.append(1)

            # Save outputs
            new_X = np.array(new_X)
            unactivated_new_X = np.array(unactivated_new_X)
            outputs.append(new_X)
            unactivated_outputs.append(unactivated_new_X)

            # Update output for next loop
            X = new_X

        return (outputs, unactivated_outputs)

    # Convenience function.  Returns final network outputs only
    def predict(self, X):
        
        output, unactivated = self.full_predict(X)
        return output[len(output)-1]

    # Train this network
    def train(self, X_train, Y_train, learning_rate, n_epochs):

        # Expect an NxM numpy array, where N is number of training instances, and M is number of inputs
        (N, M) = X_train.shape

        # Set batch size.  TODO: Implement mini-batch descent
        batch_size = N

        # Reset training history
        self.train_hist = np.zeros(shape=(n_epochs, self.n_outputs))

        # Check for correct array input size
        if M != self.n_inputs:
            raise ValueError("Training data must be an NxM numpy array, where M is the number of network inputs.")

        # For each epoch...
        for epoch in range(n_epochs):

            # Print status if requested
            if self.print_output:
                print("Training epoch ", epoch+1, " of ", n_epochs, "...")

            # Initialize squared-error sum for this epoch
            err = np.zeros(shape=(1, self.n_outputs))

            # Initialize gradients to zero, in the same shape as the weights.
            # Will sum gradients through the training batch
            grad = [None]*len(self.weights)
            for k in range(len(self.weights)):
                row, col = self.weights[k].shape
                grad[k] = np.zeros(shape=(row, col))

            # For each training sample
            for n in range(N):

                # Run a prediction
                outputs, unactivated_outputs = self.full_predict(X_train[n, :])
                y = outputs[len(outputs)-1]

                # Calc deltas for output layer
                delta = [None]*(self.get_num_layers()-1)
                delta[len(delta)-1] = np.array(y-Y_train[n])

                # Calc error for this training instance
                err = err + (y-Y_train[n])**2

                # Backpropagate
                for k in range(self.get_num_layers()-2, -1, -1):

                    # Update weights based on previous deltas
                    (J, I) = self.weights[k].shape

                    for i in range(I):  # For every neuron we are coming from...
                        for j in range(J):  # For each neuron we are going to

                            # Add to the running gradient total
                            grad[k][j, i] = grad[k][j, i]+delta[k][j]*outputs[k][i]

                    # Only need to calculate delta's if we are ahead of the input layer.
                    if k > 0:

                        # Extract forward weights and number of neurons
                        forward_weights = self.weights[k]
                        num_neurons = self.layers[k].get_neuron_count()

                        temp = []
                        # Loop through neurons, plus one for bias
                        for j in range(num_neurons+1):

                            # First extract aj, the unactivated input coming into this neuron
                            aj = unactivated_outputs[k][j]

                            # Loop through the weights feeding forward and summate weight*delta
                            this_sum = 0
                            for p in range(len(forward_weights)):
                                this_forward_weight = forward_weights[p][j]
                                this_delta = delta[k][p]
                                this_sum += this_forward_weight*this_delta

                            derivative = self.layers[k].activation_derivative(aj)
                            delta_j = derivative * this_sum
                            temp.append(delta_j)

                        delta[k-1] = np.array(temp)

            # Now we have summated gradients. Compute and save updated weights
            for k in range(len(self.weights)):
                row, col = self.weights[k].shape
                for R in range(row):
                    for C in range(col):
                        self.weights[k][R, C] = self.weights[k][R, C] - grad[k][R, C]*learning_rate/batch_size

            # Convert squared-error sum to average
            err = err/batch_size

            # Save error history over training
            self.train_hist[epoch, :] = err

    # Save this network
    def save_network(self, fn):
        fileObject = open(fn, 'wb')
        pkl.dump(self, fileObject)
        fileObject.close()

    # Load a network
    @staticmethod
    def load_network(fn):
        fileObject = open(fn, 'rb')
        network = pkl.load(fileObject)
        fileObject.close()
        return network


class Layer:

    def __init__(self, n_neurons, activation=None):

        self.n_neurons = n_neurons
        if activation is None:
            self.activate = Activation.NoActivation()
        else:
            self.activate = activation

    # Returns neuron count for this layer. DOES NOT include bias.
    def get_neuron_count(self):
        return self.n_neurons

    def activation(self, x):
        return self.activate.f(x)

    def activation_derivative(self, x):
        return self.activate.dfdx(x)

    def xavier_weight_distrib(self, n_inputs, n_outputs):
        return self.activate.xavier_weight_distrib(n_inputs, n_outputs)


class Activation:

    class Sigmoid:

        def f(self, x):
            return 1 / (1 + math.exp(-x))

        def dfdx(self, x):
            return (1 / (1 + math.exp(-x)))*(1-(1 / (1 + math.exp(-x))))

        # Returns the Xavier intialization parameter for a uniform distrib
        def xavier_weight_distrib(self, n_inputs, n_outputs):
            return math.sqrt(6/(n_inputs+n_outputs))

    class NoActivation:

        def f(self, x):
            return x

        def dfdx(self, x):
            return 1

        # For no activation function, assume uniform distrib of [-1, 1]
        def xavier_weight_distrib(self, n_inputs, n_outputs):
            return 1

    class Relu:

        def f(self, x):
            return max(0, x)

        def dfdx(self, x):
            if x > 0:
                return 1
            else:
                return 0

        # Returns the Xavier intialization parameter for a uniform distrib
        def xavier_weight_distrib(self, n_inputs, n_outputs):
            return math.sqrt(2)*math.sqrt(6 / (n_inputs + n_outputs))

    # Leaky ReLU takes the hyperparameter alpha, the slope for negative inputs
    class LeakyRelu:

        def __init__(self, alpha):
            self.alpha = alpha

        def f(self, x):
            if x > 0:
                return x
            else:
                return self.alpha*x

        def dfdx(self, x):
            if x > 0:
                return 1
            else:
                return self.alpha

        # Returns the Xavier intialization parameter for a uniform distrib
        def xavier_weight_distrib(self, n_inputs, n_outputs):
            return math.sqrt(2)*math.sqrt(6 / (n_inputs + n_outputs))

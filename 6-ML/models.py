import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.get_weights())
        

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        score = nn.as_scalar(nn.DotProduct(x, self.get_weights()))
        return 1 if score >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:
            all_correct = True
            for feature, label in dataset.iterate_once(1):
                pred = self.get_prediction(feature)
                if pred != nn.as_scalar(label):
                    all_correct = False
                    nn.Parameter.update(self.get_weights(), feature, nn.as_scalar(label))
            if all_correct:
                break
            
                
        

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.weights_1 = nn.Parameter(1, 100)
        self.bias_1 = nn.Parameter(1, 100)
        self.weights_2 = nn.Parameter(100, 1)
        self.bias_2 = nn.Parameter(1, 1)
        self.learning_rate = -0.005
        

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        result_1 = nn.AddBias(nn.Linear(x, self.weights_1), self.bias_1)
        activated_1 = nn.ReLU(result_1)
        result_2 = nn.AddBias(nn.Linear(activated_1, self.weights_2), self.bias_2)
        # activated_2 = nn.ReLU(result_2)
        # return activated_2
        return result_2
        

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        loss_item = float('inf')
        while loss_item > 0.015:
            for x, y in dataset.iterate_once(1):
                loss = self.get_loss(x, y)
                loss_item = nn.as_scalar(loss)
                grad_w1, grad_b1, grad_w2, grad_b2 = nn.gradients(loss, [self.weights_1, self.bias_1, self.weights_2, self.bias_2])
                self.weights_1.update(grad_w1, self.learning_rate)
                self.bias_1.update(grad_b1, self.learning_rate)
                self.weights_2.update(grad_w2, self.learning_rate)
                self.bias_2.update(grad_b2, self.learning_rate)

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.weights_1 = nn.Parameter(784, 1000)
        self.bias_1 = nn.Parameter(1, 1000)
        self.weights_2 = nn.Parameter(1000, 500)
        self.bias_2 = nn.Parameter(1, 500)
        self.weights_3 = nn.Parameter(500, 10)
        self.bias_3 = nn.Parameter(1, 10)
        self.learning_rate = -0.07
        self.batch_size = 100

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        result_1 = nn.AddBias(nn.Linear(x, self.weights_1), self.bias_1)
        activated_1 = nn.ReLU(result_1)
        result_2 = nn.AddBias(nn.Linear(activated_1, self.weights_2), self.bias_2)
        activated_2 = nn.ReLU(result_2)
        result_3 = nn.AddBias(nn.Linear(activated_2, self.weights_3), self.bias_3)
        return result_3

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while dataset.get_validation_accuracy() < 0.973:
            for features, labels in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(features, labels)
                grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3 = nn.gradients(loss, [self.weights_1, self.bias_1, self.weights_2, self.bias_2, self.weights_3, self.bias_3])
                self.weights_1.update(grad_w1, self.learning_rate)
                self.bias_1.update(grad_b1, self.learning_rate)
                self.weights_2.update(grad_w2, self.learning_rate)
                self.bias_2.update(grad_b2, self.learning_rate)
                self.weights_3.update(grad_w3, self.learning_rate)
                self.bias_3.update(grad_b3, self.learning_rate)
            print("Validation accuracy: ", dataset.get_validation_accuracy())
             

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # initial param
        self.hidden_size = 400
        self.learning_rate = -0.15
        self.initial_weights = nn.Parameter(self.num_chars, self.hidden_size)
        self.initial_bias = nn.Parameter(1, self.hidden_size)
        self.hidden_weights_1 = nn.Parameter(self.hidden_size, self.hidden_size)
        self.hidden_bias_1 = nn.Parameter(1, self.hidden_size)
        self.hidden_weights_2 = nn.Parameter(self.hidden_size, self.hidden_size)
        self.hidden_bias_2 = nn.Parameter(1, self.hidden_size)
        self.output_weights = nn.Parameter(self.hidden_size, len(self.languages))
        self.output_bias = nn.Parameter(1, len(self.languages))
        self.batch_size = 100
        
        

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        hidden = nn.AddBias(nn.Linear(xs[0], self.initial_weights), self.initial_bias)
        for i in range(1, len(xs)):
            hidden = nn.Add(nn.Linear(xs[i], self.initial_weights), nn.Linear(hidden, self.hidden_weights_1))
            hidden = nn.AddBias(hidden, self.hidden_bias_1)
            hidden = nn.ReLU(hidden)
            hidden = nn.Linear(hidden, self.hidden_weights_2)
            hidden = nn.AddBias(hidden, self.hidden_bias_2)
            hidden = nn.ReLU(hidden)
        result = nn.AddBias(nn.Linear(hidden, self.output_weights), self.output_bias)
        return result
            

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        pred = self.run(xs)
        return nn.SoftmaxLoss(pred, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        epoch = 0
        while epoch <= 23 or dataset.get_validation_accuracy() < 0.83:
            epoch += 1
            for features, labels in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(features, labels)
                grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3, grad_w4, grad_b4 = nn.gradients(loss, [self.initial_weights, self.initial_bias, self.hidden_weights_1, self.hidden_bias_1, self.hidden_weights_2, self.hidden_bias_2, self.output_weights, self.output_bias])
                self.initial_weights.update(grad_w1, self.learning_rate)
                self.initial_bias.update(grad_b1, self.learning_rate)
                self.hidden_weights_1.update(grad_w2, self.learning_rate)
                self.hidden_bias_1.update(grad_b2, self.learning_rate)
                self.hidden_weights_2.update(grad_w3, self.learning_rate)
                self.hidden_bias_2.update(grad_b3, self.learning_rate)
                self.output_weights.update(grad_w4, self.learning_rate)
                self.output_bias.update(grad_b4, self.learning_rate)
            

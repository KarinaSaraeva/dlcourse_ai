import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    if len(predictions.shape) == 1:
      normalised_predictions=predictions - np.amax(predictions)
      probs = np.exp(normalised_predictions)/np.sum(np.exp(normalised_predictions))
    if len(predictions.shape) == 2:
      normalised_predictions=predictions - np.amax(predictions, axis = 1)[:, np.newaxis]
      probs = np.exp(normalised_predictions)/(np.sum(np.exp(normalised_predictions), axis = 1)[:, np.newaxis])
    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    if len(probs.shape) == 1:
      loss = -np.log(probs[target_index])
    if len(probs.shape) == 2:
      target_index = target_index.reshape(target_index.shape[0])
      loss = np.mean(-np.log(np.choose(target_index, probs.T)))
    return  loss


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient
 
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    dprediction = np.copy(probs)
    if len(probs.shape) == 1:
      dprediction[target_index] -= 1
    if len(probs.shape) == 2:
      target_index = target_index.reshape(target_index.shape[0])
      dprediction[np.arange(len(target_index)), target_index] -= 1
      dprediction /= len(target_index)
    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    loss = reg_strength * np.sum(W**2)
    grad = reg_strength * 2*W
    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)
    dW = np.dot(X.transpose(), dprediction)
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100,
            epochs=1, learning_rate=1e-7, reg=1e-5, output=False):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''
        self.learning_rate = learning_rate
        self.reg = reg

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in tqdm(range(epochs)):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            loss = 0
            for batche_indices in batches_indices:
              batch = X[batche_indices]
              batch_y = y[batche_indices]
              sm_loss, sm_dW = linear_softmax(batch, self.W, batch_y)
              l2_loss, l2_dW = l2_regularization(self.W, self.reg)
              loss += sm_loss + l2_loss
              dW = sm_dW + l2_dW
              self.W -= self.learning_rate * dW
             # end
            loss_history.append(loss)
            if (output):
              print("Epoch %i, loss: %f" % (epoch, loss))
        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        y_pred = np.argmax(np.dot(X, self.W), axis = 1)
        return y_pred



                
                                                          

            

                

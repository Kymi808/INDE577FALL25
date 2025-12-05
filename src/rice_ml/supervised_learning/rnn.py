"""
Recurrent Neural Network implementations.

Simple NumPy-only RNN and GRU for sequence classification.
"""

import numpy as np


def softmax(x):
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def tanh(x):
    return np.tanh(x)


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


class RNNClassifier:
    """
    Simple RNN for sequence classification.
    
    Parameters
    ----------
    input_size : int
        Size of input at each timestep (e.g., alphabet size).
    hidden_size : int
        Number of hidden units.
    output_size : int
        Number of classes.
    learning_rate : float
        Learning rate for SGD.
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.hidden_size = hidden_size
        self.lr = learning_rate
        
        # Xavier initialization
        scale = lambda fan_in, fan_out: np.sqrt(2.0 / (fan_in + fan_out))
        
        self.Wxh = np.random.randn(input_size, hidden_size) * scale(input_size, hidden_size)
        # Identity-like initialization for Whh helps with gradient flow
        self.Whh = np.eye(hidden_size) * 0.5 + np.random.randn(hidden_size, hidden_size) * 0.1
        self.bh = np.zeros(hidden_size)
        
        self.Why = np.random.randn(hidden_size, output_size) * scale(hidden_size, output_size)
        self.by = np.zeros(output_size)
    
    def forward(self, xs):
        """
        Forward pass through sequence.
        
        Parameters
        ----------
        xs : list of ndarray
            List of one-hot input vectors, each shape (batch, input_size).
            
        Returns
        -------
        probs : ndarray
            Output probabilities, shape (batch, output_size).
        cache : dict
            Cached values for backprop.
        """
        batch_size = xs[0].shape[0]
        h = np.zeros((batch_size, self.hidden_size))
        hs = [h]
        
        for x in xs:
            h = tanh(x @ self.Wxh + h @ self.Whh + self.bh)
            hs.append(h)
        
        logits = h @ self.Why + self.by
        probs = softmax(logits)
        
        return probs, {'xs': xs, 'hs': hs}
    
    def backward(self, probs, y, cache):
        """
        Backward pass (BPTT).
        
        Parameters
        ----------
        probs : ndarray
            Predicted probabilities.
        y : ndarray
            One-hot targets.
        cache : dict
            Cached forward values.
        """
        xs, hs = cache['xs'], cache['hs']
        batch_size = y.shape[0]
        
        # Output gradients
        dlogits = (probs - y) / batch_size
        dWhy = hs[-1].T @ dlogits
        dby = dlogits.sum(axis=0)
        dh = dlogits @ self.Why.T
        
        # RNN gradients (BPTT)
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)
        
        for t in reversed(range(len(xs))):
            dtanh = dh * (1 - hs[t+1] ** 2)
            dWxh += xs[t].T @ dtanh
            dWhh += hs[t].T @ dtanh
            dbh += dtanh.sum(axis=0)
            dh = dtanh @ self.Whh.T
        
        # Gradient clipping
        for grad in [dWxh, dWhh, dbh, dWhy, dby]:
            np.clip(grad, -10, 10, out=grad)
        
        # Update weights
        self.Wxh -= self.lr * dWxh
        self.Whh -= self.lr * dWhh
        self.bh -= self.lr * dbh
        self.Why -= self.lr * dWhy
        self.by -= self.lr * dby
    
    def train_step(self, xs, y):
        """Single training step. Returns loss."""
        probs, cache = self.forward(xs)
        loss = -np.mean(np.sum(y * np.log(probs + 1e-15), axis=1))
        self.backward(probs, y, cache)
        return loss
    
    def predict(self, xs):
        """Predict class labels."""
        probs, _ = self.forward(xs)
        return np.argmax(probs, axis=1)


class GRUClassifier:
    """
    GRU for sequence classification.
    
    Parameters
    ----------
    input_size : int
        Size of input at each timestep.
    hidden_size : int
        Number of hidden units.
    output_size : int
        Number of classes.
    learning_rate : float
        Learning rate for SGD.
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.hidden_size = hidden_size
        self.lr = learning_rate
        
        scale = lambda f_in, f_out: np.sqrt(2.0 / (f_in + f_out))
        
        # Update gate - initialize bias to small positive to start with more updating
        self.Wz = np.random.randn(input_size + hidden_size, hidden_size) * scale(input_size + hidden_size, hidden_size)
        self.bz = np.ones(hidden_size) * 0.5
        
        # Reset gate - initialize bias to small positive  
        self.Wr = np.random.randn(input_size + hidden_size, hidden_size) * scale(input_size + hidden_size, hidden_size)
        self.br = np.ones(hidden_size) * 0.5
        
        # Candidate
        self.Wh = np.random.randn(input_size + hidden_size, hidden_size) * scale(input_size + hidden_size, hidden_size)
        self.bh = np.zeros(hidden_size)
        
        # Output
        self.Why = np.random.randn(hidden_size, output_size) * scale(hidden_size, output_size)
        self.by = np.zeros(output_size)
    
    def forward(self, xs):
        """Forward pass."""
        batch_size = xs[0].shape[0]
        h = np.zeros((batch_size, self.hidden_size))
        
        cache = {'xs': xs, 'hs': [h], 'zs': [], 'rs': [], 'h_tildes': []}
        
        for x in xs:
            xh = np.hstack([x, h])
            z = sigmoid(xh @ self.Wz + self.bz)
            r = sigmoid(xh @ self.Wr + self.br)
            xrh = np.hstack([x, r * h])
            h_tilde = tanh(xrh @ self.Wh + self.bh)
            h = (1 - z) * h + z * h_tilde
            
            cache['hs'].append(h)
            cache['zs'].append(z)
            cache['rs'].append(r)
            cache['h_tildes'].append(h_tilde)
        
        logits = h @ self.Why + self.by
        probs = softmax(logits)
        
        return probs, cache
    
    def backward(self, probs, y, cache):
        """Backward pass."""
        xs = cache['xs']
        batch_size = y.shape[0]
        
        dlogits = (probs - y) / batch_size
        dWhy = cache['hs'][-1].T @ dlogits
        dby = dlogits.sum(axis=0)
        dh = dlogits @ self.Why.T
        
        dWz = np.zeros_like(self.Wz)
        dWr = np.zeros_like(self.Wr)
        dWh = np.zeros_like(self.Wh)
        dbz = np.zeros_like(self.bz)
        dbr = np.zeros_like(self.br)
        dbh = np.zeros_like(self.bh)
        
        for t in reversed(range(len(xs))):
            h_prev = cache['hs'][t]
            z, r = cache['zs'][t], cache['rs'][t]
            h_tilde = cache['h_tildes'][t]
            x = xs[t]
            
            dh_tilde = dh * z
            dz = dh * (h_tilde - h_prev)
            dh_prev = dh * (1 - z)
            
            # Candidate gradient
            dtanh = dh_tilde * (1 - h_tilde ** 2)
            xrh = np.hstack([x, r * h_prev])
            dWh += xrh.T @ dtanh
            dbh += dtanh.sum(axis=0)
            dxrh = dtanh @ self.Wh.T
            dr_from_h = dxrh[:, x.shape[1]:] * h_prev
            dh_prev += dxrh[:, x.shape[1]:] * r
            
            # Gate gradients
            xh = np.hstack([x, h_prev])
            
            dsig_z = dz * z * (1 - z)
            dWz += xh.T @ dsig_z
            dbz += dsig_z.sum(axis=0)
            dh_prev += (dsig_z @ self.Wz.T)[:, x.shape[1]:]
            
            dsig_r = (dr_from_h) * r * (1 - r)
            dWr += xh.T @ dsig_r
            dbr += dsig_r.sum(axis=0)
            dh_prev += (dsig_r @ self.Wr.T)[:, x.shape[1]:]
            
            dh = dh_prev
        
        # Clip and update
        for g in [dWz, dWr, dWh, dbz, dbr, dbh, dWhy, dby]:
            np.clip(g, -10, 10, out=g)
        
        self.Wz -= self.lr * dWz
        self.Wr -= self.lr * dWr
        self.Wh -= self.lr * dWh
        self.bz -= self.lr * dbz
        self.br -= self.lr * dbr
        self.bh -= self.lr * dbh
        self.Why -= self.lr * dWhy
        self.by -= self.lr * dby
    
    def train_step(self, xs, y):
        """Single training step. Returns loss."""
        probs, cache = self.forward(xs)
        loss = -np.mean(np.sum(y * np.log(probs + 1e-15), axis=1))
        self.backward(probs, y, cache)
        return loss
    
    def predict(self, xs):
        """Predict class labels."""
        probs, _ = self.forward(xs)
        return np.argmax(probs, axis=1)
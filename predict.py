from model import lmmNet
import numpy as np
import tensorflow as tf

def predict_fn(x, t, model):
    """
    Define the derivatives learned by ML
    I think this is the best implementation, more robust than flatten()
    
    Args:
    x -- values for the current time point
    t -- time, dummy argument to conform with scipy's API
    model -- the learned ML model
    """
    dat = x.reshape(1,-1)
    dat = tf.convert_to_tensor(dat, dtype=tf.float32)
    return np.ravel(model.predict(dat))

def compute_MSE(pred, data, index):
    """
    A simple function to compute the L2 norm between pred and data
    Data and predictions are assumed to be multi-dimensional
    
    Args:
    * pred -- the predictions
    * data -- the ground truth
    * index -- the component index of the multi-dimensional data to evaluate
    """
    pred_array = np.array(pred)
    data = np.squeeze(data)
    return np.linalg.norm(data[:,index] - pred_array[:,index], 2)/np.linalg.norm(data[:,index], 2)

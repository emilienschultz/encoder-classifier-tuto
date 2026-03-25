from gc import collect as gc_collect
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch import Tensor
from torch.cuda import empty_cache, synchronize, ipc_collect
from torch.cuda import is_available as cuda_available
from torch.nn import Sigmoid
from transformers import EvalPrediction
import numpy as np

def multi_label_metrics(results_matrix, labels : Tensor, threshold : float = 0.5
                        ) -> dict:
    '''Taking a results matrix (batch_size x num_labels), the function (with a 
    threshold) associates labels to the results => y_pred
    From this y_pred matrix, evaluate the f1_micro, roc_auc and accuracy metrics
    '''
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = Sigmoid()
    probs = sigmoid(Tensor(results_matrix))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    return {'f1_micro': f1_micro_average,
            'f1_macro': f1_macro_average,
             'roc_auc': roc_auc,
             'accuracy': accuracy}

def compute_metrics(model_output):
    if isinstance(model_output.predictions,tuple):
        results_matrix = model_output.predictions[0]
    else:
        results_matrix = model_output.predictions

    metrics = multi_label_metrics(results_matrix=results_matrix, 
        labels=model_output.label_ids)
    return metrics

def clean_memory():
  """Flush GPU memory"""
  empty_cache()
  if cuda_available():
      synchronize()
      ipc_collect()
  gc_collect()
  print("Memory flushed")

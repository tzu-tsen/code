import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def plot_confusion_matrix(cm, class_names, normalize=True):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.
  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  figure = plt.figure(figsize=(8, 8))
  
  # Compute the labels from the normalized confusion matrix.
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # cm = np.around(cm*100,2) #*100後，取小數點後2位  
    cm = np.around(cm,4) #取小數點後4位          
  else:
    plt.colorbar()

  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")  
  tick_marks = range(len(class_names))
  # tick_marks = npgettext.arange(len(class_names))
  # 设置x轴坐标label
  # plt.xticks(tick_marks, class_names, rotation=45)
  plt.xticks(tick_marks, class_names, rotation=90)
  # 设置y轴坐标label
  plt.yticks(tick_marks, class_names)
  
  # # Use white text if squares are dark; otherwise black.  
  threshold = cm.max() / 2
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    # plt.text(j, i, labels[i, j], horizontalalignment="center")
    plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
  
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure

def figure_confusion_matrix(trues, preds, 
  plt_class_names=['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt'],
  normalize=True):    
  # Calculate the confusion matrix.
  sklearn_class_names=[i for i in range(len(plt_class_names))]  
  cm = confusion_matrix(trues, preds, labels=sklearn_class_names)
  # print(cm)
  figure = plot_confusion_matrix(cm, class_names=plt_class_names, normalize=normalize)

  return figure
  # writer.add_figure(
  #   'Confusion Matrix', 
  #   figure
  # )
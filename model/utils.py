import os
import pickle
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
from .preprocess import pca
import matplotlib.pyplot as plt
import torch





def plot_weight_value(alpha, label, modality1='mRNA', modality2='protein'):
  """\
  Plotting weight values
  
  """  
  import pandas as pd  
  
  df = pd.DataFrame(columns=[modality1, modality2, 'label'])  
  df[modality1], df[modality2] = alpha[:, 0], alpha[:, 1]
  df['label'] = label
  df = df.set_index('label').stack().reset_index()
  df.columns = ['label_SpatialGlue', 'Modality', 'Weight value']
  ax = sns.violinplot(data=df, x='label_SpatialGlue', y='Weight value', hue="Modality",
                split=True, inner="quart", linewidth=1, show=False)
  ax.set_title(modality1 + ' vs ' + modality2) 

  plt.tight_layout(w_pad=0.05)
  plt.show()


def add_noise_by_zeroing(matrix, zero_prob):
    """
    Zero out a fraction of the entries of the given matrix as noise.

    Parameters:
    - matrix (torch.Tensor): input matrix.
    - zero_prob (float): probability of zeroing each entry, in [0, 1].

    Returns:
    - the noised matrix.
    """
    # build a mask with the same shape as the input, where each entry survives with
    # probability 1 - zero_prob
    noise_mask = torch.bernoulli((1 - zero_prob) * torch.ones_like(matrix)).to(matrix.device)
    # apply the mask, zeroing the sampled entries
    noisy_matrix = matrix * noise_mask
    return noisy_matrix


def add_noise_by_zeroing_columns(matrix, zero_prob=0.1):
    """
    Randomly select a fraction of the columns of the given matrix and zero out the
    whole column as noise.

    Parameters:
    - matrix (torch.Tensor): input matrix.
    - zero_prob (float): probability of zeroing a column, in [0, 1].

    Returns:
    - the noised matrix.
    """
    # get the shape of the matrix
    num_columns = matrix.size(1)  # number of columns

    # draw random column indices deciding which columns to zero out
    zero_mask = torch.bernoulli((1 - zero_prob) * torch.ones(num_columns)).to(matrix.device)

    # build a mask with the same shape as the matrix
    column_mask = zero_mask.unsqueeze(0).expand_as(matrix)  # broadcast the column mask to the matrix shape

    # zero out the selected columns
    noisy_matrix = matrix * column_mask
    return noisy_matrix


def add_gaussian_noise(matrix, mean=0.0, std=0.001):
    """
    Add element-wise Gaussian noise to the given matrix.

    Parameters:
    - matrix (torch.Tensor): input matrix.
    - mean (float): mean of the Gaussian noise.
    - std (float): standard deviation of the Gaussian noise.

    Returns:
    - the noised matrix.
    """
    # draw noise with the same shape as the input
    noise = torch.normal(mean=mean, std=std, size=matrix.size()).to(matrix.device)
    # add the noise to the matrix
    noisy_matrix = matrix + noise
    return noisy_matrix

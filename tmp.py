import torch.nn as nn
import torch
from convLSTM import ConvLSTMCell


rnn = ConvLSTMCell(728, 1, (3, 3), True)
print(rnn)

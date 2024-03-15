import batch8
import batch16
import rnn
import batch64
import batch128
import batch256

import matplotlib.pyplot as plt

plt.subplot(2, 3, 1)
plt.plot(batch8.preds, label="prediction")
plt.plot(batch8.dataset.label[30:], label="actual")
plt.title('batch_size=8')

plt.subplot(2, 3, 2) 
plt.plot(batch16.preds, label="prediction")
plt.plot(batch16.dataset.label[30:], label="actual")
plt.title('batch_size=16')

plt.subplot(2, 3, 3)
plt.plot(rnn.preds, label="prediction")
plt.plot(rnn.dataset.label[30:], label="actual")
plt.title('batch_size=32')

plt.subplot(2, 3, 4) 
plt.plot(batch64.preds, label="prediction")
plt.plot(batch64.dataset.label[30:], label="actual")
plt.title('batch_size=64')

plt.subplot(2, 3, 5)
plt.plot(batch128.preds, label="prediction")
plt.plot(batch128.dataset.label[30:], label="actual")
plt.title('batch_size=128')

plt.subplot(2, 3, 6) 
plt.plot(batch256.preds, label="prediction")
plt.plot(batch256.dataset.label[30:], label="actual")
plt.title('batch_size=256')

plt.show()
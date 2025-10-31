import mnist
import numpy as np
import urllib.error
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax


# The mnist package handles the MNIST dataset for us!
# Learn more at https://github.com/datapythonista/mnist
# We only use the first 1k testing examples (out of 10k total)
# in the interest of time. Feel free to change this if you want.

try:
    # Try to load MNIST via the `mnist` package. This may attempt to download
    # the dataset if it's not already present which will fail when offline.
    train_images = mnist.train_images()[:1000]
    train_labels = mnist.train_labels()[:1000]
    test_images = mnist.test_images()[:1000]
    test_labels = mnist.test_labels()[:1000]
except (urllib.error.URLError, urllib.error.HTTPError) as e:
    # We're either offline or the URL is no longer valid.
    # Fall back to synthetic data so the demo can run.
    print('Warning: could not download MNIST dataset ({}). Using synthetic data for demo.'.format(str(e)))
    # Generate synthetic training and test data
    train_images = np.random.randint(0, 256, size=(1000, 28, 28)).astype(np.float32)
    train_labels = np.random.randint(0, 10, size=(1000,))
    test_images = np.random.randint(0, 256, size=(1000, 28, 28)).astype(np.float32)
    test_labels = np.random.randint(0, 10, size=(1000,))

conv = Conv3x3(8)                  # 28x28x1 -> 26x26x8
pool = MaxPool2()                  # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10) # 13x13x8 -> 10

def forward(image, label):
  '''
  Completes a forward pass of the CNN and calculates the accuracy and
  cross-entropy loss.
  - image is a 2d numpy array
  - label is a digit
  '''
  # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
  # to work with. This is standard practice.
  out = conv.forward((image / 255) - 0.5)
  out = pool.forward(out)
  out = softmax.forward(out)

  # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
  loss = -np.log(out[label])
  acc = 1 if np.argmax(out) == label else 0

  return out, loss, acc

def train(im, label, lr=.005):
  '''
  Completes a full training step on the given image and label.
  Returns the cross-entropy loss and accuracy.
  - image is a 2d numpy array
  - label is a digit
  - lr is the learning rate
  '''
  # Forward
  out, loss, acc = forward(im, label)

  # Calculate initial gradient
  gradient = np.zeros(10)
  gradient[label] = -1 / out[label]

  # Backprop
  gradient = softmax.backprop(gradient, lr)
  gradient = pool.backprop(gradient)
  gradient = conv.backprop(gradient, lr)

  return loss, acc

# print('MNIST CNN initialized!')

print('MNIST CNN initialized!')

# Train the CNN for 3 epochs
for epoch in range(3):
  print('--- Epoch %d ---' % (epoch + 1))

  # Shuffle the training data
  permutation = np.random.permutation(len(train_images))
  train_images = train_images[permutation]
  train_labels = train_labels[permutation]

loss = 0
num_correct = 0
for i, (im, label) in enumerate(zip(test_images, test_labels)):
  # Do a forward pass.
  _, l, acc = forward(im, label)
  loss += l
  num_correct += acc

  # Print stats every 100 steps.
  if i % 100 == 99:
    print(
      '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
      (i + 1, loss / 100, num_correct)
    )
    loss = 0
    num_correct = 0
    
  l, acc = train(im, label)
  loss += l
  num_correct += acc

# Test the CNN
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
  _, l, acc = forward(im, label)
  loss += l
  num_correct += acc

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)

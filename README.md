deformation image registration:

Training:

Manually set target\_folder and save\_dir in train.py, and set download=True when loading MNIST dataset if needed. Then run train.py to train the model. Hyperparameters such as middle_channels, epochs, learning rate, weight decay, log steps can also be set manually.

Evaluation:

Manually set model path and target path in eval.py and run eval.py to evaluate the SSIM score on the test set.\\

import numpy as np

def generate_xor_data():
    features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([[0], [1], [1], [0]])
    return features, labels

class LearningRateScheduler:
    def __init__(self, initial_lr=0.01, decay_type='step', step_size=10, decay_factor=0.5):
        self.initial_lr = initial_lr
        self.decay_type = decay_type
        self.step_size = step_size
        self.decay_factor = decay_factor
        self.current_lr = initial_lr

    def get_lr(self, epoch):
        if self.decay_type == 'step':
            self.current_lr = self.initial_lr * (self.decay_factor ** (epoch // self.step_size))
        elif self.decay_type == 'exponential':
            self.current_lr = self.initial_lr * np.exp(-self.decay_factor * epoch)
        elif self.decay_type == 'time':
            self.current_lr = self.initial_lr / (1 + self.decay_factor * epoch)
        else:
            self.current_lr = self.initial_lr
        return self.current_lr

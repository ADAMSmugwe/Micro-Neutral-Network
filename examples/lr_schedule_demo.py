import numpy as np
import matplotlib.pyplot as plt

from src.network import Network
from src.layers import Layer
from src.utils import generate_xor_data, LearningRateScheduler

if __name__ == "__main__":
    X, y = generate_xor_data()

    # Fixed learning rate
    net_fixed = Network(
        layers=[Layer(2, 4, 'tanh'), Layer(4, 1, 'sigmoid')],
        reg_lambda=0.0
    )
    net_fixed.set_loss("mse")
    fixed_history = net_fixed.train(
        X, y, epochs=2000, lr=0.1, momentum=0.9, batch_size=4, print_every=500
    )

    # Step decay
    scheduler = LearningRateScheduler(initial_lr=0.1, decay_type='step', step_size=500, decay_factor=0.5)
    net_decay = Network(
        layers=[Layer(2, 4, 'tanh'), Layer(4, 1, 'sigmoid')],
        reg_lambda=0.0
    )
    net_decay.set_loss("mse")
    decay_history = net_decay.train(
        X, y, epochs=2000, lr=0.1, momentum=0.9, batch_size=4, print_every=500, lr_scheduler=scheduler
    )

    plt.plot(fixed_history, label='Fixed LR')
    plt.plot(decay_history, label='Step Decay')
    plt.xlabel('Epoch (x500)')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Learning Rate Schedule Comparison')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

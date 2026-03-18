import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.network import Network
from src.layers import Layer


def compute_numerical_gradient(net, X, y, param_getter, idx, eps=1e-5):
    param = param_getter()
    original_val = param.flat[idx]

    param.flat[idx] = original_val + eps
    loss_plus = net.loss(y, net.forward(X))

    param.flat[idx] = original_val - eps
    loss_minus = net.loss(y, net.forward(X))

    param.flat[idx] = original_val
    return (loss_plus - loss_minus) / (2 * eps)


def gradient_check():
    np.random.seed(42)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    def mse_loss_sum(y_true, y_pred):
        return np.sum((y_true - y_pred) ** 2)

    def mse_derivative_sum(y_true, y_pred):
        return 2 * (y_pred - y_true)

    net_sum = Network()
    net_sum.add_layer(Layer(2, 3, 'tanh', dropout_rate=0.0))
    net_sum.add_layer(Layer(3, 1, 'sigmoid', dropout_rate=0.0))
    net_sum.loss_func = mse_loss_sum
    net_sum.loss_derivative = mse_derivative_sum
    net_sum.eval_mode()

    y_pred_sum = net_sum.forward(X)
    net_sum.backward(y, y_pred_sum)

    layer = net_sum.layers[0]
    pname, grad_name = 'weights', 'dW'
    grad = getattr(layer, grad_name)
    idx = 0
    num_grad = compute_numerical_gradient(
        net_sum, X, y,
        lambda l=layer, p=pname: getattr(l, p),
        idx
    )
    analytical_grad = grad.flat[idx]
    rel_diff = np.abs(analytical_grad - num_grad) / (np.abs(analytical_grad) + np.abs(num_grad) + 1e-8)
    print(f"[SUM LOSS] analytical={analytical_grad:.6e}, numerical={num_grad:.6e}, rel diff={rel_diff:.2e}")

    net = Network()
    net.add_layer(Layer(2, 3, 'tanh', dropout_rate=0.0))
    net.add_layer(Layer(3, 1, 'sigmoid', dropout_rate=0.0))
    net.set_loss('mse')
    net.eval_mode()

    y_pred = net.forward(X)
    net.backward(y, y_pred)

    num_checks = 5
    for li, layer in enumerate(net.layers):
        for pname, grad_name in [('weights', 'dW'), ('biases', 'db')]:
            grad = getattr(layer, grad_name)
            for _ in range(num_checks):
                idx = np.random.randint(getattr(layer, pname).size)
                num_grad = compute_numerical_gradient(
                    net, X, y,
                    lambda l=layer, p=pname: getattr(l, p),
                    idx
                )
                analytical_grad = grad.flat[idx]
                rel_diff = np.abs(analytical_grad - num_grad) / (np.abs(analytical_grad) + np.abs(num_grad) + 1e-8)
                print(f"Layer {li} {pname} idx {idx}: analytical={analytical_grad:.6e}, numerical={num_grad:.6e}, rel diff={rel_diff:.2e}")
                assert rel_diff < 1e-4, f"Gradient check failed for {pname} at idx {idx} in layer {li}"

    print("Gradient check passed for all tested parameters!")


if __name__ == '__main__':
    gradient_check()

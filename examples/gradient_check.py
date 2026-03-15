import numpy as np
from src.network import Network
from src.layers import Layer

def compute_numerical_gradient(net, X, y, param_getter, param_setter, idx, eps=1e-5):
    param = param_getter()
    original_val = param.flat[idx]
    # +eps
    param.flat[idx] = original_val + eps
    loss_plus = net.loss(y, net.forward(X))
    # -eps
    param.flat[idx] = original_val - eps
    loss_minus = net.loss(y, net.forward(X))
    # restore
    param.flat[idx] = original_val
    print(f"    [DEBUG] idx={idx}, loss+={loss_plus:.6e}, loss-={loss_minus:.6e}, diff={(loss_plus - loss_minus):.6e}")
    return (loss_plus - loss_minus) / (2 * eps)

def gradient_check():
    # Tiny XOR dataset
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])

    # Test with summed loss for diagnosis
    def mse_loss_sum(y_true, y_pred):
        return np.sum((y_true - y_pred) ** 2)
    def mse_derivative_sum(y_true, y_pred):
        return 2 * (y_pred - y_true)

    print("[DEBUG] Testing with summed loss (no averaging)...")
    net_sum = Network()
    net_sum.add_layer(Layer(2, 3, 'tanh', dropout_rate=0.0))
    net_sum.add_layer(Layer(3, 1, 'sigmoid', dropout_rate=0.0))
    net_sum.loss_func = mse_loss_sum
    net_sum.loss_derivative = mse_derivative_sum
    net_sum.eval_mode()
    y_pred_sum = net_sum.forward(X)
    net_sum.backward(y, y_pred_sum)
    # Check one parameter for summed loss
    layer = net_sum.layers[0]
    pname, grad_name = 'weights', 'dW'
    param = getattr(layer, pname)
    grad = getattr(layer, grad_name)
    idx = 0
    num_grad = compute_numerical_gradient(
        net_sum, X, y,
        lambda: getattr(layer, pname),
        lambda v: setattr(layer, pname, v),
        idx
    )
    analytical_grad = grad.flat[idx]
    print(f"[SUM LOSS] analytical_grad={analytical_grad:.6e}, numerical_grad={num_grad:.6e}")
    rel_diff = np.abs(analytical_grad - num_grad) / (np.abs(analytical_grad) + np.abs(num_grad) + 1e-8)
    print(f"[SUM LOSS] rel diff={rel_diff:.2e}")
    # Set random seed for reproducibility
    np.random.seed(42)
    # Tiny XOR dataset
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])
    # Simple network
    net = Network()
    net.add_layer(Layer(2, 3, 'tanh', dropout_rate=0.0))
    net.add_layer(Layer(3, 1, 'sigmoid', dropout_rate=0.0))
    net.set_loss('mse')
    # Set all layers to eval mode (disable dropout, batch norm updates)
    net.eval_mode()
    # Forward/backward
    y_pred = net.forward(X)
    net.backward(y, y_pred)
    # Check multiple random weights and biases in each layer
    num_checks = 5
    for li, layer in enumerate(net.layers):
        for pname, grad_name in [('weights', 'dW'), ('biases', 'db')]:
            param = getattr(layer, pname)
            grad = getattr(layer, grad_name)
            for _ in range(num_checks):
                idx = np.random.randint(param.size)
                num_grad = compute_numerical_gradient(
                    net, X, y,
                    lambda: getattr(layer, pname),
                    lambda v: setattr(layer, pname, v),
                    idx
                )
                analytical_grad = grad.flat[idx]
                print(f"    [DEBUG] analytical_grad={analytical_grad:.6e}, numerical_grad={num_grad:.6e}")
                rel_diff = np.abs(analytical_grad - num_grad) / (np.abs(analytical_grad) + np.abs(num_grad) + 1e-8)
                print(f"Layer {li} {pname} idx {idx}: analytical={analytical_grad:.6e}, numerical={num_grad:.6e}, rel diff={rel_diff:.2e}")
                assert rel_diff < 1e-4, f"Gradient check failed for {pname} at idx {idx} in layer {li}"
    print("Gradient check passed for all tested parameters!")

if __name__ == '__main__':
    gradient_check()

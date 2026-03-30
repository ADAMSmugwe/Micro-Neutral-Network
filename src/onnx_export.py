"""ONNX exporter for the Micro-Neural-Network framework.

The exported model expects **NCHW** input (the standard ONNX/ONNX-Runtime
convention).  Because our layers were trained with **NHWC** data, a
``Transpose`` node (NCHW → NHWC) is automatically inserted before ``Flatten``
so that the Dense-layer weights remain valid without any modification.

Usage::

    from src.onnx_export import ONNXExporter

    net.eval_mode()
    exporter = ONNXExporter(net, input_shape=(1, 28, 28))   # C, H, W
    exporter.export('mnist_cnn.onnx', output_classes=10)

    # Run with ONNX Runtime (transpose your NHWC inputs first):
    x_nchw = x_nhwc.transpose(0, 3, 1, 2)
    preds  = ONNXExporter.run('mnist_cnn.onnx', x_nchw)
"""

import numpy as np


class ONNXExporter:
    """Walk a :class:`~src.network.Network` and emit an ONNX graph.

    Supported layer types
    ---------------------
    Conv2D, ReLU, MaxPool2D, Flatten, Layer (dense), BatchNorm
    """

    # ── init ────────────────────────────────────────────────────────────────

    def __init__(self, model, input_shape):
        """
        Parameters
        ----------
        model : Network
            Trained network to export (call ``eval_mode()`` first).
        input_shape : tuple of int
            Shape of a *single* input sample in **CHW** order, e.g. ``(1, 28, 28)``.
        """
        self.model = model
        self.input_shape = tuple(input_shape)   # (C, H, W)
        self._nodes = []
        self._initializers = []
        self._counts = {}

    # ── helpers ──────────────────────────────────────────────────────────────

    def _uid(self, prefix):
        """Return a unique tensor / node name."""
        n = self._counts.get(prefix, 0) + 1
        self._counts[prefix] = n
        return f'{prefix}_{n}'

    def _init(self, array, name):
        """Register a weight array as an ONNX initializer."""
        from onnx import numpy_helper
        self._initializers.append(
            numpy_helper.from_array(array.astype(np.float32), name=name)
        )

    # ── layer converters ─────────────────────────────────────────────────────

    def _conv2d(self, layer, inp):
        from onnx import helper
        w_name = self._uid('conv_w')
        b_name = self._uid('conv_b')
        out    = self._uid('conv')

        # Our filter shape: (H, W, C_in, C_out)
        # ONNX Conv weight shape: (C_out, C_in, H, W)
        self._init(layer.filters.transpose(3, 2, 0, 1), w_name)
        self._init(layer.biases.reshape(-1), b_name)

        self._nodes.append(helper.make_node(
            'Conv', [inp, w_name, b_name], [out],
            kernel_shape=[layer.filter_size, layer.filter_size],
            strides=[layer.stride, layer.stride],
            pads=[layer.padding] * 4,
        ))
        return out

    def _relu(self, inp):
        from onnx import helper
        out = self._uid('relu')
        self._nodes.append(helper.make_node('Relu', [inp], [out]))
        return out

    def _maxpool(self, layer, inp):
        from onnx import helper
        out = self._uid('pool')
        self._nodes.append(helper.make_node(
            'MaxPool', [inp], [out],
            kernel_shape=[layer.pool_size, layer.pool_size],
            strides=[layer.stride, layer.stride],
        ))
        return out

    def _flatten(self, inp):
        """Transpose NCHW → NHWC, then flatten (axis=1).

        The Transpose preserves the element ordering the Dense weights expect,
        since those weights were learned with NHWC-flattened activations.
        """
        from onnx import helper
        nhwc = self._uid('nhwc')
        flat = self._uid('flat')
        self._nodes.append(
            helper.make_node('Transpose', [inp], [nhwc], perm=[0, 2, 3, 1])
        )
        self._nodes.append(
            helper.make_node('Flatten', [nhwc], [flat], axis=1)
        )
        return flat

    def _dense(self, layer, inp):
        """Export a Dense (Layer) as Gemm + optional activation node."""
        from onnx import helper
        w_name  = self._uid('gemm_w')
        b_name  = self._uid('gemm_b')
        gemm_out = self._uid('gemm')

        # Store weights as (out, in); use transB=1 so ONNX computes X @ W.T + b
        self._init(layer.weights.T, w_name)
        self._init(layer.biases.reshape(-1), b_name)
        self._nodes.append(helper.make_node(
            'Gemm', [inp, w_name, b_name], [gemm_out],
            alpha=1.0, beta=1.0, transB=1,
        ))

        act = layer.activation
        if act == 'relu':
            out = self._uid('relu')
            self._nodes.append(helper.make_node('Relu', [gemm_out], [out]))
            return out
        if act == 'softmax':
            out = self._uid('softmax')
            self._nodes.append(helper.make_node('Softmax', [gemm_out], [out], axis=1))
            return out
        if act == 'sigmoid':
            out = self._uid('sigmoid')
            self._nodes.append(helper.make_node('Sigmoid', [gemm_out], [out]))
            return out
        if act == 'tanh':
            out = self._uid('tanh')
            self._nodes.append(helper.make_node('Tanh', [gemm_out], [out]))
            return out
        return gemm_out   # linear — no extra node

    def _batchnorm(self, layer, inp):
        """Export BatchNorm in inference mode (running stats frozen)."""
        from onnx import helper
        scale_name = self._uid('bn_scale')
        bias_name  = self._uid('bn_bias')
        mean_name  = self._uid('bn_mean')
        var_name   = self._uid('bn_var')
        out        = self._uid('bn')

        self._init(layer.gamma.reshape(-1),        scale_name)
        self._init(layer.beta.reshape(-1),         bias_name)
        self._init(layer.running_mean.reshape(-1), mean_name)
        self._init(layer.running_var.reshape(-1),  var_name)

        self._nodes.append(helper.make_node(
            'BatchNormalization',
            [inp, scale_name, bias_name, mean_name, var_name],
            [out],
            epsilon=float(layer.eps),
            momentum=float(layer.momentum),
        ))
        return out

    # ── public API ───────────────────────────────────────────────────────────

    def export(self, filepath, output_classes=None):
        """Build and save the ONNX model.

        Parameters
        ----------
        filepath : str
            Destination ``.onnx`` file path.
        output_classes : int, optional
            Number of output classes.  Inferred from the last layer when omitted.

        Returns
        -------
        str
            The *filepath* that was saved.
        """
        import onnx
        from onnx import helper, TensorProto, checker

        in_name  = 'input'
        input_vi = helper.make_tensor_value_info(
            in_name, TensorProto.FLOAT,
            [None] + list(self.input_shape),   # [batch, C, H, W]
        )

        current = in_name
        for layer in self.model.layers:
            t = type(layer).__name__
            if   t == 'Conv2D':    current = self._conv2d(layer, current)
            elif t == 'ReLU':      current = self._relu(current)
            elif t == 'MaxPool2D': current = self._maxpool(layer, current)
            elif t == 'Flatten':   current = self._flatten(current)
            elif t == 'Layer':     current = self._dense(layer, current)
            elif t == 'BatchNorm': current = self._batchnorm(layer, current)
            else:
                print(f'  Warning: layer "{t}" not supported for ONNX export — skipped.')

        # Infer output_classes from the last Dense layer
        if output_classes is None:
            last = self.model.layers[-1]
            output_classes = last.weights.shape[1] if hasattr(last, 'weights') else None

        out_shape = [None, output_classes] if output_classes else [None]
        output_vi = helper.make_tensor_value_info(current, TensorProto.FLOAT, out_shape)

        graph = helper.make_graph(
            self._nodes, 'micro_nn',
            [input_vi], [output_vi],
            self._initializers,
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid('', 13)],
        )
        model.ir_version = 8

        checker.check_model(model)
        onnx.save(model, filepath)
        print(f'  Saved → {filepath}')
        return filepath

    @staticmethod
    def run(onnx_path, x_nchw):
        """Run inference with ONNX Runtime.

        Parameters
        ----------
        onnx_path : str
            Path to the ``.onnx`` file.
        x_nchw : np.ndarray
            Input batch in **NCHW** format (transpose from NHWC if needed).

        Returns
        -------
        np.ndarray
            Raw output of shape ``(batch, classes)``.
        """
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        in_name = sess.get_inputs()[0].name
        return sess.run(None, {in_name: x_nchw.astype(np.float32)})[0]

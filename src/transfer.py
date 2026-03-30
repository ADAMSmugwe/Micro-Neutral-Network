import numpy as np
from .layers import Layer


class TransferLearning:
    """Adapt a pretrained Network to a new task.

    Supports two strategies:
      - Feature extraction: freeze all base layers, replace only the classifier.
      - Fine-tuning: unfreeze the last few parametrized layers and train with a
        lower learning rate.

    Usage::

        tl = TransferLearning(base_model, new_output_size=10)
        model = tl.prepare_for_feature_extraction()
        # train model ...
        tl.prepare_for_fine_tuning(n_unfreeze=3)
        # train model again with lower lr ...
    """

    def __init__(self, base_model, new_output_size):
        self.base_model = base_model
        self.new_output_size = new_output_size

    def prepare_for_feature_extraction(self):
        """Freeze all base layers and replace the final classifier layer.

        The last layer of the base model must be a Dense (Layer) layer.
        Returns the modified model ready for training.
        """
        # Freeze every parametrized layer in the base model
        for layer in self.base_model.layers:
            if hasattr(layer, 'freeze'):
                layer.freeze()

        # Drop the old classifier (last layer must be a Dense Layer)
        last = self.base_model.layers[-1]
        if not hasattr(last, 'weights'):
            raise ValueError(
                "Last layer has no weights — expected a Dense (Layer) classifier."
            )
        feature_size = last.weights.shape[0]   # input dimension of old classifier
        self.base_model.layers = self.base_model.layers[:-1]

        # Attach a fresh classifier for the new task
        new_classifier = Layer(feature_size, self.new_output_size, 'softmax')
        self.base_model.layers.append(new_classifier)
        return self.base_model

    def prepare_for_fine_tuning(self, n_unfreeze=3):
        """Unfreeze the last *n_unfreeze* parametrized layers for fine-tuning.

        Call this after :meth:`prepare_for_feature_extraction` and an initial
        feature-extraction training phase.  Use a much lower learning rate
        (typically 10x smaller) to avoid destroying pretrained representations.

        Parameters
        ----------
        n_unfreeze : int
            Number of layers (counting from the end, including the new
            classifier) to unfreeze.  Defaults to 3.
        """
        # Collect layers that have freeze/unfreeze (i.e. parametrized layers)
        param_layers = [l for l in self.base_model.layers if hasattr(l, 'freeze')]

        for i, layer in enumerate(param_layers):
            if i >= len(param_layers) - n_unfreeze:
                layer.unfreeze()
            else:
                layer.freeze()

        return self.base_model

    def freeze_all(self):
        """Freeze every parametrized layer in the model."""
        for layer in self.base_model.layers:
            if hasattr(layer, 'freeze'):
                layer.freeze()
        return self.base_model

    def unfreeze_all(self):
        """Unfreeze every parametrized layer in the model."""
        for layer in self.base_model.layers:
            if hasattr(layer, 'unfreeze'):
                layer.unfreeze()
        return self.base_model

    def frozen_count(self):
        """Return (frozen, total) count of parametrized layers."""
        param_layers = [l for l in self.base_model.layers if hasattr(l, 'trainable')]
        frozen = sum(1 for l in param_layers if not l.trainable)
        return frozen, len(param_layers)

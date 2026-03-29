"""
Day 25 – Knowledge Distillation
================================
Train a small "student" network to mimic a larger "teacher" network by
learning from the teacher's soft probability distributions rather than
only from hard (one-hot) ground-truth labels.

Core idea
---------
  Loss = α · CE(student_probs, true_labels)
       + (1-α) · T² · KL(teacher_soft ‖ student_soft)

where T is the temperature (higher → softer distributions → more
information transferred between teacher and student).

Public API
----------
  softmax_temperature(logits, temperature)
  distillation_loss(student_logits, teacher_logits, true_labels, T, α)
  distillation_loss_grad(student_logits, teacher_logits, true_labels, T, α)
  DistillationTrainer(teacher, student, temperature, alpha)
"""

import numpy as np
from .network import Network


# ── helpers ───────────────────────────────────────────────────────────────────

def softmax_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically-stable softmax with temperature scaling.

    Parameters
    ----------
    logits      : (batch, classes) array of raw scores
    temperature : > 1 softens the distribution; < 1 sharpens it
    """
    z = logits / temperature
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


# ── loss ──────────────────────────────────────────────────────────────────────

def distillation_loss(
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    true_labels: np.ndarray,
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> float:
    """Combined knowledge-distillation loss (scalar).

    Parameters
    ----------
    student_logits : (batch, C) – raw logits from the student network
    teacher_logits : (batch, C) – raw logits from the teacher network
    true_labels    : (batch, C) – one-hot encoded ground-truth labels
    temperature    : softening temperature T (typically 2–8)
    alpha          : weight for the hard-label cross-entropy component

    Returns
    -------
    Scalar loss value.

    Notes
    -----
    The soft KL term is scaled by T² so that its magnitude stays comparable
    to the hard cross-entropy term as temperature varies.
    """
    batch_size = len(true_labels)
    eps = 1e-12

    # Hard loss – standard cross-entropy with ground-truth labels
    student_probs = softmax_temperature(student_logits)
    hard_loss = -np.sum(true_labels * np.log(np.clip(student_probs, eps, 1.0))) / batch_size

    # Soft loss – KL divergence between teacher and student at temperature T
    #   KL(teacher ‖ student) = Σ teacher_soft · log(teacher_soft / student_soft)
    student_soft = softmax_temperature(student_logits, temperature)
    teacher_soft = softmax_temperature(teacher_logits, temperature)
    kl_loss = (
        np.sum(teacher_soft * (np.log(teacher_soft + eps) - np.log(student_soft + eps)))
        / batch_size
    )

    return alpha * hard_loss + (1.0 - alpha) * (temperature ** 2) * kl_loss


def distillation_loss_grad(
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    true_labels: np.ndarray,
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> np.ndarray:
    """Gradient of distillation_loss w.r.t. student_logits.

    Derivation
    ----------
    ∂(hard_loss)/∂z_student  =  α · (softmax(z) - y) / N
    ∂(soft_loss)/∂z_student  =  (1-α)·T · (softmax(z/T) - softmax(z_teacher/T)) / N

    Combined:
        grad = α·(p - y)/N  +  (1-α)·T·(p_soft - q_soft)/N
    """
    batch_size = len(true_labels)

    # Hard term
    student_probs = softmax_temperature(student_logits)
    grad_hard = alpha * (student_probs - true_labels) / batch_size

    # Soft term  (note: T² in the loss / T from the softmax chain rule = T net)
    student_soft = softmax_temperature(student_logits, temperature)
    teacher_soft = softmax_temperature(teacher_logits, temperature)
    grad_soft = (1.0 - alpha) * temperature * (student_soft - teacher_soft) / batch_size

    return grad_hard + grad_soft


# ── trainer ───────────────────────────────────────────────────────────────────

class DistillationTrainer:
    """Train a student Network with knowledge distillation from a teacher.

    Parameters
    ----------
    teacher     : pre-trained Network (weights are never updated)
    student     : Network to be trained
    temperature : distillation temperature (default 4.0)
    alpha       : weight for hard cross-entropy loss (default 0.7)

    Usage
    -----
    trainer = DistillationTrainer(teacher, student, temperature=4.0, alpha=0.7)
    history = trainer.train(X_train, y_train, epochs=30,
                            val_data=(X_val, y_val))
    """

    def __init__(
        self,
        teacher: Network,
        student: Network,
        temperature: float = 4.0,
        alpha: float = 0.7,
    ):
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        self.teacher.eval_mode()   # teacher is always in inference mode

    # ------------------------------------------------------------------
    def train_step(
        self,
        X_batch: np.ndarray,
        y_batch: np.ndarray,
        lr: float,
        optimizer: str = "adam",
        momentum: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> float:
        """Single batch update. Returns the distillation loss for this batch."""
        # Student forward – returns logits (pre-softmax)
        student_logits = self.student.forward_logits(X_batch)

        # Teacher forward (inference only)
        self.teacher.eval_mode()
        teacher_logits = self.teacher.forward_logits(X_batch)

        # Compute loss and gradient
        loss = distillation_loss(
            student_logits, teacher_logits, y_batch,
            self.temperature, self.alpha,
        )
        grad = distillation_loss_grad(
            student_logits, teacher_logits, y_batch,
            self.temperature, self.alpha,
        )

        # Backward through student
        self.student._backward_from_grad(grad)

        # L2 regularisation (if configured on the student)
        if self.student.reg_lambda > 0:
            for layer in self.student.layers:
                if hasattr(layer, 'weights') and layer.dW is not None:
                    layer.dW += self.student.reg_lambda * layer.weights

        self.student.update(lr, momentum, optimizer=optimizer,
                            beta1=beta1, beta2=beta2, eps=eps)
        return float(loss)

    # ------------------------------------------------------------------
    def train_epoch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.001,
        batch_size: int = 64,
        optimizer: str = "adam",
        momentum: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> float:
        """One full pass over the training data. Returns mean batch loss."""
        self.student.train_mode()
        indices = np.random.permutation(len(X))
        total_loss, n_batches = 0.0, 0

        for i in range(0, len(X), batch_size):
            idx = indices[i : i + batch_size]
            total_loss += self.train_step(
                X[idx], y[idx], lr, optimizer, momentum, beta1, beta2, eps
            )
            n_batches += 1

        return total_loss / n_batches if n_batches else 0.0

    # ------------------------------------------------------------------
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 30,
        lr: float = 0.001,
        batch_size: int = 64,
        optimizer: str = "adam",
        val_data=None,
        verbose: bool = True,
    ) -> dict:
        """Full distillation training loop.

        Parameters
        ----------
        X_train, y_train : training data and one-hot labels
        epochs           : number of epochs
        lr               : learning rate
        batch_size       : mini-batch size
        optimizer        : 'adam' or 'sgd'
        val_data         : optional (X_val, y_val) tuple for tracking val metrics
        verbose          : print progress each epoch

        Returns
        -------
        dict with keys 'train_loss', and optionally 'val_loss', 'val_acc'
        """
        history: dict = {"train_loss": []}
        if val_data is not None:
            history["val_loss"] = []
            history["val_acc"] = []

        for epoch in range(epochs):
            train_loss = self.train_epoch(
                X_train, y_train,
                lr=lr, batch_size=batch_size, optimizer=optimizer,
            )
            history["train_loss"].append(train_loss)

            if val_data is not None:
                self.student.eval_mode()
                X_val, y_val = val_data
                val_logits = self.student.forward_logits(X_val)
                val_probs  = softmax_temperature(val_logits)
                eps = 1e-12
                val_loss = -float(
                    np.mean(np.sum(y_val * np.log(np.clip(val_probs, eps, 1.0)), axis=1))
                )
                val_acc  = float(np.mean(
                    np.argmax(val_probs, axis=1) == np.argmax(y_val, axis=1)
                ))
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                self.student.train_mode()

                if verbose:
                    print(
                        f"Epoch {epoch+1:3d} | "
                        f"distill_loss: {train_loss:.4f}  "
                        f"val_loss: {val_loss:.4f}  "
                        f"val_acc: {val_acc:.4f}"
                    )
            elif verbose:
                print(f"Epoch {epoch+1:3d} | distill_loss: {train_loss:.4f}")

        return history

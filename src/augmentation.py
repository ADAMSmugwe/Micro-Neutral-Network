import numpy as np

try:
    from scipy.ndimage import rotate as _scipy_rotate, shift as _scipy_shift, zoom as _scipy_zoom
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


class DataAugmentor:
    """
    On-the-fly data augmentation for image batches shaped (N, H, W, C).

    Parameters
    ----------
    rotation_range : float
        Max rotation in degrees (+/-). 0 to disable.
    shift_range : int
        Max pixel shift in x and y. 0 to disable.
    zoom_range : float
        Fractional zoom delta, e.g. 0.1 means zoom in [0.9, 1.1]. 0 to disable.
    horizontal_flip : bool
        Randomly flip images left-right with probability 0.5.
    """

    def __init__(self, rotation_range=15, shift_range=2, zoom_range=0.1, horizontal_flip=False):
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip

        if not _SCIPY_AVAILABLE and (rotation_range > 0 or zoom_range > 0):
            print(
                "Warning: scipy not found. Rotation and zoom augmentations are disabled. "
                "Install scipy for full augmentation support."
            )

    def apply(self, X, y):
        """
        Apply random augmentations to a batch.

        Parameters
        ----------
        X : ndarray, shape (N, H, W, C)
        y : ndarray, shape (N, ...)

        Returns
        -------
        X_aug : ndarray, same shape as X
        y     : ndarray, unchanged
        """
        X_aug = np.copy(X)
        for i in range(X_aug.shape[0]):
            img = X_aug[i]  # (H, W, C)

            if _SCIPY_AVAILABLE and self.rotation_range > 0:
                angle = np.random.uniform(-self.rotation_range, self.rotation_range)
                img = self._rotate(img, angle)

            if self.shift_range > 0:
                dx = np.random.randint(-self.shift_range, self.shift_range + 1)
                dy = np.random.randint(-self.shift_range, self.shift_range + 1)
                img = self._shift(img, dx, dy)

            if _SCIPY_AVAILABLE and self.zoom_range > 0:
                zoom_factor = np.random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
                img = self._zoom(img, zoom_factor)

            if self.horizontal_flip and np.random.rand() > 0.5:
                img = img[:, ::-1, :]

            X_aug[i] = img

        return X_aug, y

    def _rotate(self, img, angle):
        return _scipy_rotate(img, angle, axes=(0, 1), reshape=False, mode='constant', cval=0)

    def _shift(self, img, dx, dy):
        if _SCIPY_AVAILABLE:
            # img is (H, W, C); shift along (H, W) axes only
            return _scipy_shift(img, (dy, dx, 0), mode='constant', cval=0)
        else:
            # Fallback: np.roll with zero-fill at boundary
            img = np.roll(img, dy, axis=0)
            img = np.roll(img, dx, axis=1)
            if dy > 0:
                img[:dy, :, :] = 0
            elif dy < 0:
                img[dy:, :, :] = 0
            if dx > 0:
                img[:, :dx, :] = 0
            elif dx < 0:
                img[:, dx:, :] = 0
            return img

    def _zoom(self, img, zoom_factor):
        h, w, c = img.shape
        zoomed = _scipy_zoom(img, (zoom_factor, zoom_factor, 1), mode='constant', cval=0)
        zh, zw = zoomed.shape[:2]

        # Crop or pad back to original size
        out = np.zeros_like(img)
        if zoom_factor >= 1.0:
            # Zoomed image is larger; center-crop
            y0 = (zh - h) // 2
            x0 = (zw - w) // 2
            out = zoomed[y0:y0 + h, x0:x0 + w, :]
        else:
            # Zoomed image is smaller; center-pad
            y0 = (h - zh) // 2
            x0 = (w - zw) // 2
            out[y0:y0 + zh, x0:x0 + zw, :] = zoomed

        return out

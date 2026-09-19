# ai_ecg/augmentations.py

import numpy as np

class ECGAugmentations:
    """
    On‐the‐fly augmentations for 2D ECG arrays (time × leads).

    Available transforms:
      • Gaussian noise
      • Random circular time‐shift
      • Random cropping (and zero‐padding back)
      • Time‐warp (±warp_pct)
      • Baseline wander (low‐freq sine)
      • Random lead dropout
    """

    def __init__(
        self,
        noise_std: float = 0.01,
        max_shift_s: float = 0.2,
        crop_len_s: float = 2.0,
        fs: int = 250,
        warp_pct: float = 0.05,
        drop_lead_p: float = 0.1
    ):
        """
        Parameters
        ----------
        noise_std : float
            Standard deviation of additive Gaussian noise (in mV).
        max_shift_s : float
            Maximum circular shift in seconds.
        crop_len_s : float
            Length of the random crop window in seconds.
        fs : int
            Sampling frequency (Hz).
        warp_pct : float
            Fractional time‐warp (±warp_pct).
        drop_lead_p : float
            Probability of dropping any given lead (zeroing it out).
        """
        self.noise_std   = noise_std
        self.shift_pts   = int(max_shift_s * fs)
        self.crop_pts    = int(crop_len_s * fs)
        self.fs          = fs
        self.warp_pct    = warp_pct
        self.drop_lead_p = drop_lead_p

    def __call__(self, ecg: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to one ECG trace.

        Parameters
        ----------
        ecg : np.ndarray, shape (T, L)
            T timepoints × L leads.

        Returns
        -------
        aug_ecg : np.ndarray, shape (T, L)
            Augmented ECG.
        """
        T, L = ecg.shape
        x = ecg.copy()

        # 1) Additive Gaussian noise
        x += np.random.randn(T, L) * self.noise_std

        # 2) Random circular time‐shift
        if self.shift_pts > 0:
            shift = np.random.randint(-self.shift_pts, self.shift_pts)
            x = np.roll(x, shift, axis=0)

        # 3) Random crop + pad back to length T
        if self.crop_pts < T:
            start = np.random.randint(0, T - self.crop_pts + 1)
            cropped = x[start:start + self.crop_pts]
            pad_left  = start
            pad_right = T - self.crop_pts - pad_left
            x = np.pad(cropped,
                       ((pad_left, pad_right), (0, 0)),
                       mode="constant", constant_values=0.0)

        # 4) Time‐warp (linear rescale)
        warp = 1.0 + np.random.uniform(-self.warp_pct, self.warp_pct)
        xp   = np.linspace(0, T - 1, T) * warp
        # clamp xp to valid range
        xp = np.clip(xp, 0, T - 1)
        x_warp = np.zeros_like(x)
        for li in range(L):
            x_warp[:, li] = np.interp(xp, np.arange(T), x[:, li])
        x = x_warp

        # 5) Baseline wander: add low‐freq sinusoid
        freq  = np.random.uniform(0.1, 0.5)
        drift = 0.1 * np.sin(2 * np.pi * freq * np.arange(T) / self.fs)
        x += drift[:, None]

        # 6) Random lead dropout
        for li in range(L):
            if np.random.rand() < self.drop_lead_p:
                x[:, li] = 0.0

        return x

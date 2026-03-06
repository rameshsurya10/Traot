"""
Successive Variational Mode Decomposition (SVMD)
=================================================

SVMD extracts modes successively without needing to know the number of modes.
Superior to VMD and CEEMDAN for financial time series.

Key Advantages:
- No need to preset mode count (unlike VMD)
- Lower computational complexity than VMD
- More robust against initial values
- 73% RMSE reduction in stock prediction

Research: Wiley 2025 - SVMD-LSTM Hybrid Approach for Stock Market Prediction
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq
from typing import List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DecompositionResult:
    """Result of SVMD decomposition."""
    imfs: np.ndarray  # Intrinsic Mode Functions (n_modes, n_samples)
    residual: np.ndarray  # Residual after extracting all modes
    center_frequencies: List[float]  # Center frequency of each mode
    n_modes: int  # Number of modes extracted
    reconstruction_error: float  # Error in reconstructing original signal


class SVMDDecomposer:
    """
    Successive Variational Mode Decomposition.

    Unlike VMD which extracts all modes simultaneously, SVMD extracts
    modes one at a time from high to low frequency. This successive
    approach means we don't need to know the number of modes beforehand.

    Parameters:
    -----------
    alpha : float
        Bandwidth constraint parameter. Higher = narrower bandwidth.
        Default: 2000 (good for financial data)
    tau : float
        Noise tolerance parameter. 0 = no noise tolerance.
        Default: 0
    tol : float
        Convergence tolerance for each mode extraction.
        Default: 1e-7
    max_modes : int
        Maximum number of modes to extract.
        Default: 5
    min_energy_ratio : float
        Stop extracting modes when energy ratio falls below this.
        Default: 0.01
    """

    def __init__(
        self,
        alpha: float = 2000,
        tau: float = 0,
        tol: float = 1e-7,
        max_modes: int = 5,
        min_energy_ratio: float = 0.01
    ):
        self.alpha = alpha
        self.tau = tau
        self.tol = tol
        self.max_modes = max_modes
        self.min_energy_ratio = min_energy_ratio

    def decompose(self, signal: np.ndarray) -> DecompositionResult:
        """
        Decompose signal into Intrinsic Mode Functions (IMFs).

        Args:
            signal: 1D numpy array of the signal to decompose

        Returns:
            DecompositionResult containing IMFs, residual, and metadata
        """
        signal = np.asarray(signal, dtype=np.float64)
        n = len(signal)

        # Store original for reconstruction error calculation
        original = signal.copy()

        # Initialize
        residual = signal.copy()
        imfs = []
        center_frequencies = []
        original_energy = np.sum(signal ** 2)

        for mode_idx in range(self.max_modes):
            # Extract one mode
            mode, center_freq = self._extract_mode(residual)

            # Check if mode has enough energy
            mode_energy = np.sum(mode ** 2)
            energy_ratio = mode_energy / (original_energy + 1e-10)

            if energy_ratio < self.min_energy_ratio:
                logger.debug(f"Mode {mode_idx}: energy ratio {energy_ratio:.4f} below threshold, stopping")
                break

            imfs.append(mode)
            center_frequencies.append(center_freq)

            # Update residual
            residual = residual - mode

            logger.debug(f"Mode {mode_idx}: freq={center_freq:.4f}, energy_ratio={energy_ratio:.4f}")

            # Check if residual is too small
            residual_energy = np.sum(residual ** 2)
            if residual_energy / original_energy < self.min_energy_ratio:
                logger.debug("Residual energy too small, stopping")
                break

        # Convert to numpy array
        imfs = np.array(imfs) if imfs else np.array([]).reshape(0, n)

        # Calculate reconstruction error
        if len(imfs) > 0:
            reconstructed = np.sum(imfs, axis=0) + residual
            reconstruction_error = np.sqrt(np.mean((original - reconstructed) ** 2))
        else:
            reconstruction_error = 0.0

        return DecompositionResult(
            imfs=imfs,
            residual=residual,
            center_frequencies=center_frequencies,
            n_modes=len(imfs),
            reconstruction_error=reconstruction_error
        )

    def _extract_mode(self, signal: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Extract a single mode using variational optimization.

        Uses ADMM (Alternating Direction Method of Multipliers) for optimization.
        """
        n = len(signal)

        # FFT of signal
        signal_fft = fft(signal)
        freqs = fftfreq(n)

        # Initialize mode in frequency domain
        # Start with estimate based on dominant frequency
        magnitudes = np.abs(signal_fft)
        positive_mask = freqs > 0

        if np.any(positive_mask):
            dominant_idx = np.argmax(magnitudes[positive_mask])
            center_freq = freqs[positive_mask][dominant_idx]
        else:
            center_freq = 0.1

        # Initialize mode as bandpass around center frequency
        mode_fft = signal_fft.copy()

        # Iterative refinement
        lambda_dual = np.zeros(n, dtype=complex)  # Dual variable

        for iteration in range(100):
            old_mode_fft = mode_fft.copy()

            # Update mode (Wiener filter with bandwidth constraint)
            bandwidth = self.alpha * (freqs - center_freq) ** 2
            denominator = 1 + bandwidth + self.tau
            mode_fft = (signal_fft - lambda_dual / 2) / denominator

            # Update center frequency (weighted average of frequencies)
            positive_freqs = freqs[positive_mask]
            mode_magnitudes = np.abs(mode_fft[positive_mask]) ** 2
            total_power = np.sum(mode_magnitudes)

            if total_power > 0:
                center_freq = np.sum(positive_freqs * mode_magnitudes) / total_power
                center_freq = max(0.001, min(0.5, center_freq))  # Clamp to valid range

            # Update dual variable
            lambda_dual = lambda_dual + self.tau * (signal_fft - mode_fft)

            # Check convergence
            change = np.linalg.norm(mode_fft - old_mode_fft) / (np.linalg.norm(old_mode_fft) + 1e-10)
            if change < self.tol:
                break

        # Convert back to time domain
        mode = np.real(ifft(mode_fft))

        return mode, center_freq

    def decompose_dataframe(
        self,
        df: 'pd.DataFrame',
        column: str = 'close'
    ) -> Tuple['pd.DataFrame', DecompositionResult]:
        """
        Decompose a column from a DataFrame and add IMFs as new columns.

        Args:
            df: DataFrame with price data
            column: Column name to decompose

        Returns:
            DataFrame with added IMF columns and DecompositionResult
        """

        signal = df[column].values
        result = self.decompose(signal)

        # Add IMFs to DataFrame
        df_out = df.copy()
        for i, imf in enumerate(result.imfs):
            df_out[f'imf_{i+1}'] = imf
        df_out['imf_residual'] = result.residual

        return df_out, result

    def get_feature_names(self, n_modes: int = None) -> List[str]:
        """Get feature column names for IMFs."""
        if n_modes is None:
            n_modes = self.max_modes
        return [f'imf_{i+1}' for i in range(n_modes)] + ['imf_residual']


class VMDDecomposer:
    """
    Variational Mode Decomposition (fallback if SVMD has issues).

    VMD decomposes signal into K modes simultaneously.
    Requires knowing number of modes beforehand.

    Parameters:
    -----------
    K : int
        Number of modes to extract
    alpha : float
        Bandwidth constraint
    tau : float
        Noise tolerance
    """

    def __init__(self, K: int = 5, alpha: float = 2000, tau: float = 0):
        self.K = K
        self.alpha = alpha
        self.tau = tau

    def decompose(self, signal: np.ndarray) -> DecompositionResult:
        """Decompose signal into K modes simultaneously."""
        signal = np.asarray(signal, dtype=np.float64)
        n = len(signal)

        # FFT of signal
        signal_fft = fft(signal)
        freqs = fftfreq(n)

        # Initialize modes and center frequencies
        modes_fft = np.zeros((self.K, n), dtype=complex)
        center_freqs = np.linspace(0.05, 0.45, self.K)

        # Dual variable
        lambda_fft = np.zeros(n, dtype=complex)

        # ADMM iterations
        for _ in range(100):
            # Update each mode
            sum_modes = np.sum(modes_fft, axis=0)

            for k in range(self.K):
                # Other modes sum
                other_modes = sum_modes - modes_fft[k]

                # Wiener filter update
                bandwidth = self.alpha * (freqs - center_freqs[k]) ** 2
                numerator = signal_fft - other_modes + lambda_fft / 2
                denominator = 1 + 2 * self.alpha * bandwidth
                modes_fft[k] = numerator / denominator

                # Update center frequency
                positive_mask = freqs > 0
                mode_power = np.abs(modes_fft[k][positive_mask]) ** 2
                total_power = np.sum(mode_power)

                if total_power > 0:
                    center_freqs[k] = np.sum(freqs[positive_mask] * mode_power) / total_power

            # Update dual
            lambda_fft = lambda_fft + self.tau * (signal_fft - np.sum(modes_fft, axis=0))

        # Convert to time domain
        imfs = np.array([np.real(ifft(m)) for m in modes_fft])
        residual = signal - np.sum(imfs, axis=0)

        reconstruction_error = np.sqrt(np.mean((signal - np.sum(imfs, axis=0) - residual) ** 2))

        return DecompositionResult(
            imfs=imfs,
            residual=residual,
            center_frequencies=list(center_freqs),
            n_modes=self.K,
            reconstruction_error=reconstruction_error
        )

# (C) Daniel Strano and the Qrack contributors 2017-2025. All rights reserved.
#
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.

try:
    # Written by Elara (custom OpenAI GPT)
    import numpy as np

    def quantize_by_range(data, feature_indices, bits):
        """
        Discretize selected features of a dataset into binary variables using numpy.linspace binning.

        Args:
            data (np.ndarray): Input data of shape (n_samples, n_features).
            feature_indices (list[int]): Indices of features to discretize.
            bits (int): Number of bits to represent each discretized feature (e.g., 2 bits = 4 quantiles).

        Returns:
            np.ndarray: Transformed data with selected features replaced by binary bit columns.
        """
        n_samples, n_features = data.shape
        n_bins = 2**bits

        new_features = []
        for i in range(n_features):
            if i in feature_indices:
                min_val, max_val = data[:, i].min(), data[:, i].max()
                thresholds = np.linspace(min_val, max_val, n_bins + 1)[1:-1]
                bins = np.digitize(data[:, i], bins=thresholds)
                bin_bits = ((bins[:, None] & (1 << np.arange(bits)[::-1])) > 0).astype(int)
                new_features.append(bin_bits)
            else:
                new_features.append(data[:, i][:, None])  # Keep original

        return np.hstack(new_features)

except ImportError:

    def quantize_by_range(data, feature_indices, bits):
        """
        Discretize selected features of a dataset into binary variables using numpy.linspace binning.

        Args:
            data (np.ndarray): Input data of shape (n_samples, n_features).
            feature_indices (list[int]): Indices of features to discretize.
            bits (int): Number of bits to represent each discretized feature (e.g., 2 bits = 4 quantiles).

        Returns:
            np.ndarray: Transformed data with selected features replaced by binary bit columns.
        """
        raise NotImplementedError("You must have numpy installed to use quantize_by_percentile()!")

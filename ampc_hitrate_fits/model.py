import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d


class HitRateModel:
    """
    Statistical model for estimating hit rates in virtual screening experiments.

    This model combines a bivariate normal distribution (representing regular molecules)
    with an optional artifact distribution to account for false positives or outliers.
    It allows calculation of hit rates from modeled experimental outcomes given screening parameters.

    The model works with standardized docking scores: pProp values (log-scaled proportions)
    are converted to standardized scores from a standard Gaussian distribution (mean=0, std=1).
    Good docking scores are negative in the left tail (approximately -2 to -8). The minimum
    achievable score depends on library size: max pProp is bounded by log10(library_size).
    For example, a 1k-molecule library has best score ≈ -3, while a 100M-molecule library
    has best score ≈ -5.6.

    Parameters
    ----------
    input_params : dict
        Dictionary containing model parameters:
            - rho : float
                Correlation coefficient between predicted and experimental values.
            - exp_mean : float
                Mean of the experimental pKi distribution.
            - exp_std : float
                Standard deviation of the experimental pKi distribution.
            - artifact_freq : float, optional
                Frequency (fraction) of artifacts in the dataset.
            - artifact_mean : float, optional
                Mean of the artifact distribution.
            - artifact_std : float, optional
                Standard deviation of the artifact distribution.
    """

    def __init__(self, input_params):
        self.rho = input_params["rho"]
        self.exp_mean = input_params["exp_mean"]
        self.exp_std = input_params["exp_std"]

        self.artifact_freq = input_params.get("artifact_freq", 0)
        self.artifact_mean = input_params.get("artifact_mean", 0)
        self.artifact_std = input_params.get("artifact_std", 1)

        self.bivariate_std = np.sqrt(1 - self.rho**2)
        self.exp_threshold = None
        self.ihr = None
        self.ppf_lookup = self._make_mixture_ppf_lookup()

    def _make_mixture_ppf_lookup(self, grid_size=10000, x_min=-10, x_max=10):
        """Function to ease calculation of PPF for the mixture distribution using a lookup table."""
        x_vals = np.linspace(x_min, x_max, grid_size)
        cdf_vals = self._mixture_cdf(x_vals)
        return interp1d(
            cdf_vals,
            x_vals,
            kind="linear",
            bounds_error=False,
            fill_value=(x_min, x_max),
        )

    def _mixture_cdf(self, x):
        """Compute the cumulative distribution function (CDF) of the mixture distribution."""
        x = np.asarray(x)
        cdf_art = self.artifact_freq * norm.cdf(
            x, self.artifact_mean, self.artifact_std
        )
        cdf_reg = (1 - self.artifact_freq) * norm.cdf(x)
        return cdf_art + cdf_reg

    def _calculate_weight(self, standardized_score):
        """Compute the proportion of regular molecules at a given standardized docking score."""
        weight_regular = norm.pdf(standardized_score) * (1 - self.artifact_freq)
        weight_artifact = (
            norm.pdf(
                standardized_score, loc=self.artifact_mean, scale=self.artifact_std
            )
            * self.artifact_freq
        )
        total_weight = weight_regular + weight_artifact
        return weight_regular / total_weight

    def _calculate_hit_rate(self, standardized_score):
        """Compute the hit rate for a given standardized docking score."""
        mean = self.rho * standardized_score
        base_hit_rate = 1 - norm.cdf(
            self.exp_threshold, loc=mean, scale=self.bivariate_std
        )

        if self.artifact_freq <= 0:
            return base_hit_rate

        weight = self._calculate_weight(standardized_score)
        return weight * base_hit_rate + (1 - weight) * self.ihr

    def get_pprop_hr(self, pprop, definition):
        """
        Compute the hit rate for a given pProp value and experimental threshold (hit definition).

        This method converts pProp (log-scaled proportion) to a standardized docking score, which is
        a quantile value from a standard Gaussian distribution (mean=0, std=1). Good docking scores
        are negative (left tail, approximately -2 to -8 range).

        Important: The minimum achievable standardized score depends on library size:
        - Library of 1,000 molecules (max pProp ≈ 3): best score ≈ -3.0
        - Library of 100 million molecules (max pProp ≈ 8): best score ≈ -5.6
        The maximum pProp is bounded by log10(library_size).

        Parameters
        ----------
        pprop : float or array-like
            Log-scaled proportion of top-ranked molecules. pProp = -log10(fraction).
            Examples: pProp = 3 means top 1 in 1000, pProp = 6 means top 1 in 1 million.
            Can be a single float or an array of floats.
        definition : float or array-like
            Experimental pKi threshold for defining a hit.
            Can be a single float or an array of floats.

        Returns
        -------
        float or np.ndarray
            Estimated hit rate at the given pProp(s).
        """
        # Assert that pprop and definition have same length if they are arrays
        if isinstance(pprop, (list, np.ndarray)):
            pprop = np.asarray(pprop)
            definition = np.asarray(definition)
            assert pprop.shape == definition.shape, (
                "pprop and definition must have the same shape."
            )
        self.exp_threshold = (definition - self.exp_mean) / self.exp_std
        self.ihr = 1 - norm.cdf(self.exp_threshold)

        # Convert pProp to a standardized docking score using the mixture distribution.
        # pProp = -log10(fraction), so 10**(-pProp) gives the quantile (fraction of library in top pProp).
        # The ppf_lookup applies the inverse CDF to convert this quantile to a standardized score,
        # which follows a standard Gaussian (mean=0, std=1). Good scores are negative in the left tail.
        standardized_score = self.ppf_lookup(10 ** (-pprop))
        return self._calculate_hit_rate(standardized_score)

    def sample(self, pprop):
        """
        Samples an experimental pKi-value based on model.

        Args:
            pprop (float): The pProp of the sample.

        Returns:
            float: A sampled pKi value.
        """
        # Convert pProp to a standardized docking score using the mixture distribution
        standardized_score = self.ppf_lookup(10 ** (-pprop))
        # Calculute the fraction of regular molecules at this score
        weight = self._calculate_weight(standardized_score)
        # Sample either an artifact or a regular molecule based on the weight
        if np.random.rand() < weight:
            sample = np.random.normal(self.rho * standardized_score, self.bivariate_std)
        else:
            sample = np.random.normal(0, 1)
        # Convert the sample back to the original scale
        return (sample * self.exp_std) + self.exp_mean

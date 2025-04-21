import numpy as np


class ProbabilityModel:
    def __init__(self, N):
        self.N = N

    def get_prob(self, prior_symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            prior_symbols: numpy array of previous symbols

        Returns:
            tokens: numpy array of symbols in descending order of probability
            cdfs: numpy array of the cumulative probabilities of the tokens in the same order
        """
        raise NotImplementedError


# Static model - fixed probabilities of each symbol
class StaticModel(ProbabilityModel):
    def __init__(self, N, symbols, probs):
        super().__init__(N)
        # Ensure numpy arrays
        probs = np.array(probs)
        symbols = np.array(symbols)
        # sort in descending order of probability
        sorted_indices = np.argsort(-probs)
        self.symbols = symbols[sorted_indices]
        self.probs = probs[sorted_indices]

    def get_prob(self, prior_symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cdfs = np.cumsum(self.probs)
        # Ensure cdfs sum to 1
        cdfs /= cdfs[-1]
        return (np.array(self.symbols), cdfs)


# Simple adaptive model - places higher probability of symbol that appears more
class AdaptiveModel(ProbabilityModel):
    def __init__(self, N, symbols):
        super().__init__(N)
        self.symbols = symbols

    def get_prob(self, prior_symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probs = np.zeros(self.N)

        for i in range(len(probs)):
            probs[i] = np.sum(prior_symbols == self.symbols[i]) + 0.10
        probs /= probs.sum()

        combined_sort = sorted(
            zip(self.symbols, probs), key=lambda x: x[1], reverse=True
        )
        tokens = [x[0] for x in combined_sort]
        sorted_probs = [x[1] for x in combined_sort]

        cdfs = np.zeros(self.N)
        cumalative = 0
        for i in range(len(sorted_probs)):
            cumalative += sorted_probs[i]
            cdfs[i] = cumalative

        cdfs /= cdfs[-1]

        return (np.array(tokens), cdfs)

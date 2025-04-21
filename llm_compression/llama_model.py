from llama_cpp import Llama
import numpy as np
import time

from .probability_model import ProbabilityModel


class LlamaModel(ProbabilityModel):
    def __init__(
        self,
        model_path: str,
        top_p: float = 0.99,
        max_context: int = 50,
    ):
        """
        Initialize a LlamaModel.

        Parameters
        ----------
        model_path : str
            File path to the LLaMA model .gguf file.
        top_p : float, optional
            The top [0, 1] percentage of the most likely tokens to consider when computing the probability distribution. 
            Higher values will generally result in better compression for sequences that the LLM can easily predict.  
        max_context : int, optional
            The maximum number of tokens to keep in the model's context. Higher values will generally lead to better compression but slower performance.

        Raises
        ------
        ValueError
            If the provided max_context is too large for the model.
        """

        t1 = time.perf_counter()
        self.llm = Llama(
            model_path=model_path,
            n_ctx=max_context,
            logits_all=True,
            n_gpu_layers=0,
            verbose=False,
        )
        if self.llm.n_ctx() < max_context:
            raise ValueError(
                f"Provided max_context is too large for the model. Provided max_context is {max_context}, but model max context is {self.llm.n_ctx}"
            )
        t2 = time.perf_counter()
        print(f"Model loaded in {t2 - t1} seconds")
        self.N = self.llm.n_vocab()
        self.top_p = top_p
        self.cache = []
        self.max_context = max_context
        super().__init__(self.N)

    def get_prob(self, prior_symbols: np.ndarray[int]) -> tuple[np.ndarray, np.ndarray]:
        """
        Get cumalitive probability distribution of the next token given the prior tokens.

        Parameters
        ----------
        prior_symbols : np.ndarray[int]
            The sequence of prior tokens. 

        Returns
        -------
        (tokens, cdfs)
            tokens : np.ndarray[int]
                The symbols in descending order of probability.
            cdfs : np.ndarray[float]  
                The cumulative probabilities of the tokens in the same order.
        """
        print(f"Prior symbols: {len(prior_symbols)}")

        # If no prior tokens, return uniform distribution and clear cache
        if len(prior_symbols) == 0:
            self.reset()

            cumulative = 0.0
            tokens = np.zeros(self.N, dtype=np.int64)
            cdfs = np.zeros(self.N, dtype=np.float64)
            for token_id in range(self.N):
                cumulative += 1.0 / self.N
                tokens[token_id] = token_id
                cdfs[token_id] = cumulative
            return tokens, cdfs

        # If there are more symbols cached than context, clear oldest half of cache
        if len(self.cache) >= self.max_context:
            # self.reset()
            self.cache = self.cache[self.max_context // 2 :]
            self.llm.reset()
            # evaluate what is left of cache
            self.llm.eval(self.cache)
            print("Cache cleared")

        # Evaluate latest token
        t1 = time.perf_counter()
        self.llm.eval([prior_symbols[-1]])
        t2 = time.perf_counter()

        # Get logprobs for last token position
        logprobs = self.llm.eval_logits[-1].copy()
        logprobs = np.array(logprobs)
        probs = np.exp(logprobs - logprobs.max())
        probs /= probs.sum()

        t4 = time.perf_counter()

        # Get cdf distribution of 90% most likely tokens
        ts1 = time.perf_counter()
        topk = np.argsort(-probs)
        ts2 = time.perf_counter()

        tokens = np.zeros(self.N, dtype=np.int64)
        cdfs = np.zeros(self.N, dtype=np.float64)

        # Sorted probabilities
        probs_sorted = probs[topk]
        # Compute cumulative probabilities
        cum_probs = np.cumsum(probs_sorted)
        # Find cutoff index of top_p probability
        cutoff_index = np.searchsorted(cum_probs, self.top_p, side="right")
        # Get slice of topk
        topk_slice = topk[: cutoff_index + 1]
        n_topk = cutoff_index + 1

        tokens[:n_topk] = topk_slice
        cdfs[:n_topk] = cum_probs[:n_topk]

        # Add remaining tokens with uniform probability
        topk_mask = np.zeros(self.N, dtype=bool)
        topk_mask[topk_slice] = True
        remaining_tokens = np.flatnonzero(~topk_mask)

        n_remaining = len(remaining_tokens)
        if n_remaining > 0:
            uniform_prob = (1.0 - cum_probs[cutoff_index]) / n_remaining
            tokens[n_topk : n_topk + n_remaining] = remaining_tokens
            cdfs[n_topk : n_topk + n_remaining] = (
                uniform_prob * np.arange(1, n_remaining + 1) + cum_probs[cutoff_index]
            )

        t5 = time.perf_counter()

        # Cache new token
        self.cache.append(prior_symbols[-1])

        # returns sorted tokens and cdfs
        return (tokens, cdfs)

    def reset(self) -> None:
        """ Clear cache and reset LLM. Needed when starting a new compression/decrompression """
        self.cache = []
        self.llm.reset()

    def tokenize(self, text: bytes) -> list[int]:
        """
        Tokenize a string of bytes into a sequence of token IDs.

        This function is a wrapper around Llama's `tokenize` method without adding the BOS token.

        Parameters
        ----------
        text : bytes
            The string of bytes to tokenize.

        Returns
        -------
        list[int]
            A list of token IDs.
        """

        return self.llm.tokenize(text, add_bos=False)

    def detokenize(self, tokens: list[int]) -> bytes:
        """
        Convert a sequence of token IDs back into a string of bytes.

        This function is a wrapper around Llama's `detokenize` method.

        Parameters
        ----------
        tokens : list[int]
            A list of token IDs to be converted back into bytes.

        Returns
        -------
        bytes
            The original string of bytes.
        """

        return self.llm.detokenize(tokens)
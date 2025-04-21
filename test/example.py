from llm_compression import LlamaModel, encode, decode
import numpy as np

# Load llama model from gguf file
model = LlamaModel(model_path="../Llama-3.2-1B-Instruct-Q4_K_M.gguf")

# We will compress the following sentence from a random Wikipedia article
wiki_str = "The building began as a movie theater in 1973, was converted into the Jet Set nightclub in 1994, and underwent renovations in 2010 and 2015"
# The string needs to encoded in utf-8
encoded_str = wiki_str.encode("utf-8")
# We tokenize the string using our model
tokenized_str = np.asarray(model.tokenize(encoded_str))
# Check how many symbols we have
print(len(tokenized_str), " symbols")

# Encode the string using arithmetic coding
# This yields high compression
encoded_bin = encode(tokenized_str, model)
print(len(encoded_bin), " bits in encoding")

# Decode the string
decoded = decode(encoded_bin, model, len(tokenized_str))
outstr = model.detokenize(decoded)

# Print the decoded string to make sure we get the same as the original
print(outstr.decode("utf-8"))

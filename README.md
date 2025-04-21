# LLM-Compression
Arithmetic Compression using Llama LLM. This is an experimental package for using LLMs for compression. This yields high compression rates for natural language, but is too slow to practically be used for compression.

## Requirements
To use the LLM model for compression you need to install ```llama-cpp-python``` and download a llama llm model in a .gguf format. I used the model ```Llama-3.2-1B-Instruct-Q4_K_M.gguf``` in my examples, [download link](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF). This model can be run on any modern computer on the CPU. 

## Installation
First install [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), using the instructions on github. Then install this package with ```pip install llm-compression```. For information on how to use the package see the example below.

## Example 
```
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
# Check how many symbols we have (35 symbols)
print(len(tokenized_str), " symbols")

# Encode the string using arithmetic coding
# This yields high compression it is encoded into 146 bits, which is around 4.2 bits per symbol in the input string.
encoded_bin = encode(tokenized_str, model)
print(len(encoded_bin), " bits in encoding")

# Decode the string
decoded = decode(encoded_bin, model, len(tokenized_str))
outstr = model.detokenize(decoded)

# Print the decoded string to make sure we get the same as the original
print(outstr.decode("utf-8"))
```

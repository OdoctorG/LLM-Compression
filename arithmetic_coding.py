import numpy as np
import math

from probability_model import ProbabilityModel

def encode(input_arr: np.ndarray, model: ProbabilityModel):
    BIT_PRECISION = 64
    MAX = (1 << BIT_PRECISION) - 1
    HALF = 1 << (BIT_PRECISION - 1)
    QUARTER = 1 << (BIT_PRECISION - 2)
    
    low = 0
    high = MAX
    output = []
    pending_bits = 0
    k = 0

    for symbol in input_arr:
        symbols, cdfs = model.get_prob(input_arr[:k])
        
        symbol_idx = np.where(symbols == symbol)[0] #symbols.index(symbol)

        # Calculate probability bounds using floating-point CDFs
        cdf_low = cdfs[symbol_idx-1][0] if symbol_idx > 0 else 0.0
        cdf_high = cdfs[symbol_idx][0]
        
        # Convert to integer ranges with careful rounding
        range_size = high - low + 1
        new_low = low + math.floor(cdf_low * range_size)
        new_high = low + math.ceil(cdf_high * range_size) - 1
        
        low, high = new_low, new_high

        # Interval scaling and bit emission
        while True:
            if high < HALF:
                output.append(0)
                output.extend([1] * pending_bits)
                pending_bits = 0
                low <<= 1
                high = (high << 1) | 1
            elif low >= HALF:
                output.append(1)
                output.extend([0] * pending_bits)
                pending_bits = 0
                low = (low - HALF) << 1
                high = (high - HALF) << 1 | 1
            elif low >= QUARTER and high < 3 * QUARTER:
                pending_bits += 1
                low = (low - QUARTER) << 1
                high = (high - QUARTER) << 1 | 1
            else:
                break
        k += 1

    # Finalize remaining bits
    pending_bits += 1
    if low < QUARTER:
        output.append(0)
        output.extend([1] * pending_bits)
    else:
        output.append(1)
        output.extend([0] * pending_bits)

    return output

def decode(encoded_bits: np.ndarray, model: ProbabilityModel, num_symbols: int):
    import math
    import bisect

    BIT_PRECISION = 64
    MAX = (1 << BIT_PRECISION) - 1
    HALF = 1 << (BIT_PRECISION - 1)
    QUARTER = 1 << (BIT_PRECISION - 2)

    low = 0
    high = MAX
    value = 0
    decoded = []

    # Initialize value with the first 64 bits, pad with zeros if needed
    for i in range(BIT_PRECISION):
        value <<= 1
        if i < len(encoded_bits):
            value |= encoded_bits[i]

    bit_ptr = BIT_PRECISION

    for _ in range(num_symbols):
        symbols, cdfs = model.get_prob(np.array(decoded))
        # Insert 0.0 at the start for proper interval mapping
        cdf_bounds = [0.0] + list(cdfs)

        range_size = high - low + 1
        offset = value - low
        scaled_offset = offset / range_size

        # Find the symbol whose interval contains scaled_offset
        symbol_idx = bisect.bisect_right(cdf_bounds, scaled_offset) - 1
        decoded_symbol = symbols[symbol_idx]
        decoded.append(decoded_symbol)

        cdf_low = cdf_bounds[symbol_idx]
        cdf_high = cdf_bounds[symbol_idx + 1]

        new_low = low + math.floor(cdf_low * range_size)
        new_high = low + math.ceil(cdf_high * range_size) - 1
        low, high = new_low, new_high

        # Interval scaling and bit reading (mirror encoder)
        while True:
            if high < HALF:
                low <<= 1
                high = (high << 1) | 1
                value <<= 1
                if bit_ptr < len(encoded_bits):
                    value |= encoded_bits[bit_ptr]
                bit_ptr += 1
            elif low >= HALF:
                low = (low - HALF) << 1
                high = (high - HALF) << 1 | 1
                value = (value - HALF) << 1
                if bit_ptr < len(encoded_bits):
                    value |= encoded_bits[bit_ptr]
                bit_ptr += 1
            elif low >= QUARTER and high < 3 * QUARTER:
                low = (low - QUARTER) << 1
                high = (high - QUARTER) << 1 | 1
                value = (value - QUARTER) << 1
                if bit_ptr < len(encoded_bits):
                    value |= encoded_bits[bit_ptr]
                bit_ptr += 1
            else:
                break

    return decoded

# Testing
if __name__ == "__main__":
    from llama_model import LlamaModel
    from probability_model import StaticModel


    model = StaticModel(3, ['a', 'b', 'c'], [0.4, 0.3, 0.3])
    test_str = "abcabc"
    print(len(test_str), " symbols")
    
    test_arr = np.asarray([test_str[i] for i in range(len(test_str))])
    print(test_arr)
    encoded_bin = encode(test_arr, model)
    print(len(encoded_bin), " bits in encoding")
    print(encoded_bin)

    decoded = decode(encoded_bin, model, len(test_arr))
    print(decoded)
    
    print("LLM TEST")

    model = LlamaModel(0.99, max_context=50)
    wiki_str = "Weissman var på 1920-talet en av Finlands mest kända kuplettsångare och var en mycket aktiv skådespelare med både operetter och lustspel på sin repertoar. Hans skådespelarkarriär inleddes omkring 1913 och varade fram till 1930-talet. Under den tiden var han verksam vid flera teatrar och skådespelarensembler. Som kuplettsångare uppträdde han på biografer, kaféer och restauranger runt om i landet. På 1910- och 1920-talen gjorde han en stor mängd skivinspelningar och var en aktiv sångare under grammofonfebern 1929. När kuplettgenren gick ur mode på slutet av 1920-talet försökte Weissman anpassa sig till schlagermusiken, men övergav inom kort den konstnärliga banan för att ägna sig åt reklamverksamhet och diverse affärer"
    wiki_str_short = "Weissman var på 1920-talet en av Finlands mest kända kuplettsångare och var en mycket aktiv skådespelare med både operetter och lustspel på sin repertoar."
    wiki_str_short2 = "The building began as a movie theater in 1973, was converted into the Jet Set nightclub in 1994, and underwent renovations in 2010 and 2015"
    prompt = wiki_str.encode('utf-8')
    prompt_tkn = np.asarray(model.tokenize(prompt))
    print(len(prompt_tkn), " symbols")
    encoded_bin = encode(prompt_tkn, model)
    print(len(encoded_bin), " bits in encoding")
    decoded = decode(encoded_bin, model, len(prompt_tkn))
    outstr = model.detokenize(decoded)
    print(outstr.decode('utf-8'))
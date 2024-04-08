from quant.hamming import HammingLoss
from quant import QuantModel,QuantModule
from quant.adaptive_rounding import AdaRoundQuantizer
hamming = HammingLoss()
def compute_hamming_loss(block):
    hamming_loss = 0
    total_numel = 0
    for n,m in block.named_modules():
        if isinstance(m, QuantModule):
            if isinstance(m.weight_quantizer,AdaRoundQuantizer):
                hamming_loss += hamming(m.weight_quantizer.int_repr(m.weight),reduce="sum")
                total_numel += m.weight.numel()
    if total_numel == 0:
        return 0
    return hamming_loss / total_numel
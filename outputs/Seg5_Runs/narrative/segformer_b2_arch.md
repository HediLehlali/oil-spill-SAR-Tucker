### Architecture and what is compressed

**Encoder (MiT-B2).** Four stages of 3-4-6-3 transformer blocks (dim 64-128-320-512, 1-2-5-8 heads, 56²→7² tokens), fed by overlapping patch embeddings (7×7 stride-4, then 3×3 stride-2). There are no positional embeddings. Attention is *efficient (spatially-reduced) attention*: queries use all tokens, keys/values are computed from a strided convolution + LayerNorm of the tokens (reduction 8-4-2-1), which keeps stage 1 affordable.

**What Tucker touches.** The 16 attention blocks (Q/K/V/output projections, 5.93M parameters, 22 % of the model). The pre-softmax tensor `Wq_h·Wk_h^T` is applied to (tokens, reduced tokens), so the reduction conv + LayerNorm are kept untouched and the Tucker block sees the same reduced K/V tokens.

**After the attention (not compressed).** The output projection is folded into the post-softmax Tucker tensor; then residual + LayerNorm and the **Mix-FFN**: `Linear(C→4C) → 3×3 depth-wise conv → GELU → Linear(4C→C)` (12.0M parameters, 44 % of the model, i.e. 2× the attention). The depth-wise conv injects positional information. Then the **All-MLP decoder**: one linear layer per scale (C_i→768), upsample to ¼ resolution, concatenation (4×768), 1×1 conv + BN + ReLU, 1×1 classifier (3.2M parameters).

**FFN Tucker opportunity.** The two FFN matrices per block are (C×4C) and (4C×C); blocks of one stage share shapes (3/4/6/3 layers), so a layer-stacked (L, C, 4C) tensor can share factor matrices across layers. The depth-wise 3×3 conv (4C,1,3,3) is too small to matter.

### Architecture and what is compressed

**Encoder (Swin-T).** Patch embedding 4×4 (C=96) then four stages of 2-2-6-2 blocks (dim 96-192-384-768, 3-6-12-24 heads, window 7×7, alternating regular / shifted windows), with patch merging (2×2 concat + linear, halves resolution, doubles channels) between stages. Attention is local (49 tokens per window) with a learned relative-position bias.

**What Tucker touches.** The 12 window-attention blocks (8.64M parameters, 15 % of the model). The relative-position bias table and the shifted-window mask are kept and added to the Tucker logits.

**After the attention (not compressed).** Output projection (folded into the post-softmax Tucker tensor) → residual + LayerNorm → **MLP** `Linear(C→4C) → GELU → Linear(4C→C)` (17.3M parameters, 29 %) → patch merging. The four stage outputs feed **UperNet**: a pyramid-pooling module (scales 1/2/3/6) on stage 4, an FPN (1×1 laterals + top-down additions + 3×3 convs, 512 channels) and a fusion 3×3 conv over the four concatenated levels, then a 1×1 classifier. This convolutional head is 53 % of the parameters (31.4M) and is untouched; the auxiliary head of the original checkpoint is dropped.

**FFN Tucker opportunity.** FFN weights per stage are identical in shape (2/2/6/2 layers), so the six stage-3 blocks are the natural candidate for a cross-layer Tucker tensor (6, 384, 1536); the convolutional UperNet head is the larger target in this model (Tucker-2 on 3×3 convs, as in the CNN literature).

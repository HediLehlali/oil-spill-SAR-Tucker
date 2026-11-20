### Architecture and what is compressed

**Encoder.** A Swin transformer with 2-2-2-2 blocks (feature size 48 → dims 48-96-192-384, heads 3-6-12-24, window 7, patch size 2, so 112² tokens in stage 1). The model is the MONAI `SwinUNETR`, built here with `spatial_dims=2` because the SAR data are 2D; the published checkpoint (`MONAI/swin_unetr_btcv_segmentation`) is a **3D** CT model (62M parameters). The 2D network has 25.1M parameters. The Swin encoder and the skip encoders are initialised from the 3D weights by a documented 3D→2D conversion (conv kernels summed over depth, relative-position table cut at depth 0, patch-merging projections folded); all 129 compatible tensors are transferred, the UNETR decoder and the 2-class output layer start from scratch. This is therefore **not** a pretrained-on-SAR or ImageNet-style initialisation, and results should be read with that in mind.

**What Tucker touches.** The 8 window-attention blocks (1.57M parameters, only **6 %** of this model), so even an infinite attention compression ratio would shrink the whole network by only ~6 %.

**After the attention (not compressed).** Output projection (folded), residual + LayerNorm, MLP (3.1M, 12 %), patch merging; the stage outputs are then consumed by a **convolutional UNETR decoder**: residual conv skip-encoders at five scales, four up-blocks (transposed conv ×2 + skip concatenation + residual 3×3 convs) and a 1×1 output conv. These convolutions are 75 % of the parameters.

**FFN Tucker opportunity.** FFN is also small (12 %); in this architecture the conv decoder (Tucker-2 on 3×3 conv kernels, the classic CNN use of Tucker) holds most of the redundancy to exploit.

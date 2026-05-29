# Burn / neural metrics

The `nn-burn` feature adds a Burn integration layer without forcing any single model or checkpoint.

## Tensor adapter

```rust
use imq::nn::{frame_to_nchw_rgb_tensor, NormalizeRgb};

let tensor = frame_to_nchw_rgb_tensor::<B>(&frame, &device, NormalizeRgb::imagenet())?;
# Ok::<(), imq::Error>(())
```

The returned tensor has shape `[1, 3, H, W]` in NCHW order.

Supported inputs:

- `Luma8`;
- `Rgb8`, `Rgba8`;
- `Bgr8`, `Bgra8`;
- `Yuv444p8`, `Yuv422p8`, `Yuv420p8`, `Nv12`.

Other formats should be converted before neural evaluation or added to `read_rgb01`.

## BurnL2Metric

`BurnL2Metric` is a simple tensor-space MSE. It is useful as a smoke test for backends and preprocessing.

## LPIPS/DISTS-style metrics

Implement `BurnFeatureExtractor<B>` for your model wrapper:

```rust
use burn::tensor::{backend::Backend, Tensor};
use imq::nn::BurnFeatureExtractor;

struct MyExtractor;

impl<B: Backend> BurnFeatureExtractor<B> for MyExtractor {
    fn extract(&self, image: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
        // Run a pretrained Burn module and return feature maps.
        vec![image]
    }
}
```

Then:

```rust
use imq::nn::LpipsLikeMetric;

let metric = LpipsLikeMetric::<B, _>::new(device, MyExtractor);
let score = metric.compare(&reference, &distorted)?;
# Ok::<(), imq::Error>(())
```

A production LPIPS metric typically uses a fixed backbone such as VGG/AlexNet/SqueezeNet, feature normalization, and learned per-channel weights. DISTS uses structure/texture statistics over deep features. MUSIQ and CLIP-IQA-style metrics are no-reference or text/vision feature metrics and can still reuse the tensor adapter and backend selection in this module.

## Why pretrained weights are not bundled

Pretrained IQA model weights often have separate licenses, data provenance requirements, and preprocessing assumptions. `imq` therefore provides Burn-compatible hooks and examples while keeping checkpoints application-owned.

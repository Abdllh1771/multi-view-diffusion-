# Pose-Conditioned Diffusion for Novel View Synthesis (ShapeNet Cars)

This project implements a **pose-conditioned diffusion model** that generates novel views of ShapeNet cars given a single reference image and a target camera pose.

It uses:

- The **NMR ShapeNet** subset (as prepared by the DVR authors) for cars (`02958343`)
- A **U-Net–style architecture** conditioned on:
  - A **noisy target image**
  - A **reference image**
  - The **relative camera pose** between them
- An **autoregressive generation loop** to render a full 360° sequence around a car

---

## Overview

### Data Pipeline

- Dataset: `NMR_Dataset/shapenet/02958343` (car category)
- Each object has:
  - 24 rendered RGB views (`image/0000.png` – `image/0023.png`)
  - A `cameras.npz` file with per-view camera matrices `world_mat_i`
- The custom `NMRShapeNetDataset`:
  - Splits objects 90% train / 10% val
  - Randomly samples **two views per object**:
    - `ref_img`: reference image
    - `target_img`: target image
  - Loads corresponding camera matrices and computes:
    \[
      \text{pose\_rel} = T_\text{target} \cdot T_\text{ref}^{-1}
    \]
  - Returns:
    - `ref_img` (3×64×64)
    - `target_img` (3×64×64)
    - `relative_pose` (flattened 4×4 = 16-dim vector)

Images are normalized to `[-1, 1]` and resized to 64×64.

---

### Model

The core model is a **conditional U-Net** trained as a standard denoising diffusion model:

- **Input channels**:  
  `x_noisy` (noisy target image) concatenated with `ref_img` → 6 channels total
- **Conditioning signals**:
  - **Timestep embedding** (sinusoidal → MLP)
  - **Relative pose embedding** (16-D pose → MLP)
  - These are fused by simple addition and injected into residual blocks
- **Architecture**:
  - Residual blocks (`ResBlock`) with per-block embedding injection
  - 3 downsampling stages, bottleneck, and 3 upsampling stages with skip-connections
  - Final 3-channel output predicting the **noise** in the target image

The model learns to predict the noise added to `target_img` at a random diffusion step `t`.  
Loss: **MSE** between predicted noise and true noise.

---

### Training

- Timesteps: `TIMESTEPS = 1000`
- Beta schedule: linear from `0.0001` to `0.02`
- Optimizer: **AdamW** with `lr = 2e-4`
- Batch size: `64`
- Input resolution: `64×64`
- Default epochs: `20`
- Model checkpoints are saved every 5 epochs:
  - `3dim_model_epoch_5.pth`, `3dim_model_epoch_10.pth`, etc.

Training loop summary:

1. Sample a batch from `NMRShapeNetDataset`
2. Sample a random timestep `t` for each image
3. Generate noise and construct the noisy target:
   \[
   x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1-\bar{\alpha}_t} \cdot \epsilon
   \]
4. Predict noise with the conditional U-Net:
   \[
   \hat{\epsilon} = \text{model}(x_t, t, \text{ref\_img}, \text{relative\_pose})
   \]
5. Minimize `MSE(ε̂, ε)` and update the model

---

### Autoregressive Novel View Generation

After training, you can generate a rotating sequence around a car using **stochastic conditioning**:

1. Pick a **validation object** and use:
   - `start_img` = one reference frame from the validation set
   - `start_pose` = identity (or a chosen canonical pose)
2. Create a list of **target poses** by rotating around the Y-axis (e.g., 360° / 8 frames)
3. Call:

```python
sequence = generate_autoregressive_sequence(
    model,
    initial_ref_img=start_img,
    initial_pose=start_pose,
    target_poses=target_poses
)


Inside generate_autoregressive_sequence:

For each target pose:

Start from pure noise

At each reverse diffusion step:

Randomly pick one of the already known frames as the reference (stochastic conditioning)

Compute the relative pose from that reference to the current target pose

Use the conditional U-Net to denoise

The newly generated frame is added to the history and can be used to condition future frames

Finally, the script visualizes:

The original input frame

All generated frames in a single row using matplotlib

## Future Improvements

Planned / potential improvements on the model side:

- **Richer pose conditioning**
  - Replace the raw 4×4 matrix flattening with more structured pose encodings (e.g., SE(3) / Lie algebra, quaternions + translation).
  - Add a dedicated pose encoder and fuse pose features with cross-attention instead of simple MLP + addition.

- **Multi-view conditioning**
  - Condition the U-Net on multiple reference views instead of a single image.
  - Experiment with attention over a set of reference images + poses to better handle occlusions and ambiguous viewpoints.

- **3D-aware representations**
  - Incorporate an intermediate 3D representation (e.g., implicit field / NeRF-style features) and render back to 2D.
  - Explore 3D convolutions or transformers over a learned 3D feature grid to enforce cross-view consistency.

- **Improved diffusion training and sampling**
  - Use alternative beta schedules (e.g., cosine) and advanced samplers (DDIM, DPM-Solver, etc.) for faster and higher-quality sampling.
  - Switch from pure noise prediction (ε) to `v`-prediction or direct `x₀` prediction and compare stability/quality.
  - Add classifier-free guidance on pose or object identity to better control the generated view.

- **Higher-resolution generation**
  - Scale the U-Net to 128×128 or 256×256 using:
    - Multi-scale / hierarchical U-Nets, or
    - A separate super-resolution module trained on top of the 64×64 outputs.
  - Introduce skip connections across resolutions for sharper details.

- **Better loss functions**
  - Combine MSE with perceptual losses (e.g., LPIPS) for sharper and more realistic images.
  - Optionally add a lightweight adversarial head (GAN-style discriminator) for improved texture realism while keeping diffusion as the main training signal.

- **View-consistency regularization**
  - Add multi-view consistency losses: enforce that different generated views of the same object are geometrically compatible.
  - Encourage consistent features across views by using cycle-consistency (e.g., generate A→B→A and penalize deviations).

- **Category and domain generalization**
  - Extend to more ShapeNet categories (airplanes, chairs, etc.) and train a single model conditioned on category ID.
  - Evaluate how well a model trained on one category transfers to others and explore shared vs category-specific layers.

- **Quantitative evaluation pipeline**
  - Add automatic metrics on held-out views (PSNR, SSIM, LPIPS) and track them during training.
  - Optionally compute FID/KID on rendered views to compare against other novel view synthesis approaches.


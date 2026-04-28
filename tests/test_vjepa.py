import torch

from agent_1.models.vjepa.vjepa import VJEPA, Predictor


def _small_vjepa():
    """Tiny config that exercises every code path without being slow to run."""
    return VJEPA(
        patch_dim=(2, 4, 4),
        in_channels=3,
        grid_size=(2, 4, 4),
        encoder_dim=32,
        encoder_intermediate_size=64,
        encoder_num_heads=4,
        encoder_head_dim=8,
        encoder_num_layers=2,
        encoder_dim_partitions=(2, 4, 2),
        predictor_dim=16,
        predictor_intermediate_size=32,
        predictor_num_heads=2,
        predictor_head_dim=8,
        predictor_num_layers=1,
        predictor_dim_partitions=(2, 4, 2),
    )


def test_predictor_output_shape():
    """Predictor projects encoded context tokens into target slots and
    returns one embedding per `masks_pred` index, in encoder dim — so the
    JEPA loss can compare it directly against the EMA target."""
    B, K_enc, K_pred = 2, 10, 6
    encoder_dim, predictor_dim = 32, 16
    grid_size = (2, 4, 4)
    num_patches = grid_size[0] * grid_size[1] * grid_size[2]

    model = Predictor(
        encoder_dim=encoder_dim,
        predictor_dim=predictor_dim,
        intermediate_size=32,
        num_heads=2,
        head_dim=8,
        num_layers=1,
        grid_size=grid_size,
        dim_partitions=(2, 4, 2),
    )

    z = torch.randn(B, K_enc, encoder_dim)
    perm = torch.stack([torch.randperm(num_patches) for _ in range(B)])
    masks_enc = perm[:, :K_enc]
    masks_pred = perm[:, K_enc : K_enc + K_pred]

    out = model(z, masks_enc, masks_pred)
    assert out.shape == (B, K_pred, encoder_dim)


def test_vjepa_forward_returns_parallel_lists():
    """One `(z, h)` pair per mask config, each shaped `[B, K_pred, D_enc]`,
    matching the `MaskCollator` contract — the trainer iterates these in
    parallel to compute the loss."""
    B, T, C, H, W = 2, 4, 3, 16, 16
    grid_size = (2, 4, 4)
    num_patches = grid_size[0] * grid_size[1] * grid_size[2]

    model = _small_vjepa()
    x = torch.randn(B, C, T, H, W)

    masks_enc, masks_pred = [], []
    for K_enc, K_pred in [(10, 6), (8, 4)]:
        perm = torch.stack([torch.randperm(num_patches) for _ in range(B)])
        masks_enc.append(perm[:, :K_enc])
        masks_pred.append(perm[:, K_enc : K_enc + K_pred])

    zs, hs = model(x, masks_enc, masks_pred)

    assert len(zs) == len(hs) == 2
    for z, h, m_pred in zip(zs, hs, masks_pred):
        assert z.shape == h.shape == (B, m_pred.shape[1], 32)


def test_target_encoder_frozen_after_construction():
    """The EMA target must not receive gradients — only the context encoder
    and predictor train; the target follows via `update_target`."""
    model = _small_vjepa()
    for p in model.target_encoder.parameters():
        assert not p.requires_grad
    for p in model.encoder.parameters():
        assert p.requires_grad
    for p in model.predictor.parameters():
        assert p.requires_grad


def test_update_target_is_convex_combination():
    """`update_target(m)` produces `m * old_target + (1-m) * encoder` per
    parameter — the EMA invariant the momentum schedule relies on."""
    model = _small_vjepa()

    # snapshot initial target params (which equal initial encoder params via deepcopy)
    initial_target = [p.detach().clone() for p in model.target_encoder.parameters()]
    # diverge the encoder so target != encoder
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(1.0)
    new_encoder = [p.detach().clone() for p in model.encoder.parameters()]

    model.update_target(0.5)

    for old_t, new_e, p in zip(initial_target, new_encoder, model.target_encoder.parameters()):
        expected = 0.5 * old_t + 0.5 * new_e
        assert torch.allclose(p, expected, atol=1e-6)


def test_vjepa_gradient_flow():
    """Loss gradients reach `encoder` and `predictor` but not `target_encoder` —
    so the trainer's two Muon optimizers cover every learnable parameter."""
    B, T, C, H, W = 2, 4, 3, 16, 16
    grid_size = (2, 4, 4)
    num_patches = grid_size[0] * grid_size[1] * grid_size[2]
    K_enc, K_pred = 10, 6

    model = _small_vjepa()
    x = torch.randn(B, C, T, H, W)
    perm = torch.stack([torch.randperm(num_patches) for _ in range(B)])
    masks_enc = [perm[:, :K_enc]]
    masks_pred = [perm[:, K_enc : K_enc + K_pred]]

    zs, hs = model(x, masks_enc, masks_pred)
    loss = sum((z - h).abs().mean() for z, h in zip(zs, hs))
    loss.backward()

    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.predictor.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())

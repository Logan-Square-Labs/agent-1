import torch
from agent_1.models.vjepa.mask import MaskCollator, MaskGenerator


def test_mask_generator_shapes_and_dtype():
    """Generator returns two `[B, K]` int64 index tensors so downstream
    `torch.gather`-style mask application works without further casting."""
    grid_size = (8, 10, 9)
    batch_size = 4

    gen = MaskGenerator(
        grid_size=grid_size,
        mask_area_ratio=0.4,
        mask_ar_range=(0.75, 1.5),
        num_sub_masks=2,
    )
    masks_enc, masks_pred = gen(batch_size)

    # both are [B, K] index tensors
    assert masks_enc.ndim == 2 and masks_enc.shape[0] == batch_size
    assert masks_pred.ndim == 2 and masks_pred.shape[0] == batch_size
    assert masks_enc.dtype == torch.int64
    assert masks_pred.dtype == torch.int64


def test_mask_generator_indices_in_bounds():
    """Every index points to a valid position in the flattened
    `T * H * W` patch grid — no off-by-one from block placement."""
    grid_size = (8, 10, 9)
    num_patches = grid_size[0] * grid_size[1] * grid_size[2]

    gen = MaskGenerator(
        grid_size=grid_size,
        mask_area_ratio=0.4,
        mask_ar_range=(0.75, 1.5),
        num_sub_masks=2,
    )
    masks_enc, masks_pred = gen(batch_size=4)

    # all indices reference a valid patch in the flattened grid
    assert masks_enc.min() >= 0 and masks_enc.max() < num_patches
    assert masks_pred.min() >= 0 and masks_pred.max() < num_patches


def test_mask_generator_enc_pred_disjoint():
    """The encoder and predictor index sets share no patches per sample —
    this is the V-JEPA invariant that the predictor only ever sees targets
    the encoder did not."""
    grid_size = (8, 10, 9)

    gen = MaskGenerator(
        grid_size=grid_size,
        mask_area_ratio=0.4,
        mask_ar_range=(0.75, 1.5),
        num_sub_masks=2,
    )
    masks_enc, masks_pred = gen(batch_size=4)

    # encoder keeps patches where mask==1, predictor targets where mask==0 — never the same patch
    for i in range(masks_enc.shape[0]):
        enc_set = set(masks_enc[i].tolist())
        pred_set = set(masks_pred[i].tolist())
        assert enc_set.isdisjoint(pred_set)


def test_mask_generator_keeps_some_patches():
    """The resample loop guarantees neither side collapses to zero — without
    this, a degenerate sample (full mask or no mask) would produce empty
    index tensors that crash downstream gather/scatter ops."""
    grid_size = (8, 10, 9)

    gen = MaskGenerator(
        grid_size=grid_size,
        mask_area_ratio=0.4,
        mask_ar_range=(0.75, 1.5),
        num_sub_masks=2,
    )
    masks_enc, masks_pred = gen(batch_size=4)

    # neither side should be empty for reasonable configs (the resample loop guarantees this)
    assert masks_enc.shape[1] > 0
    assert masks_pred.shape[1] > 0


def test_mask_collator_output_keys_and_shapes():
    """The collator produces one `(masks_enc, masks_pred)` pair per entry in
    `mask_configs`, preserving order, alongside a normally-stacked video
    tensor — this is the contract the training loop depends on."""
    grid_size = (8, 10, 9)
    cfgs = [
        {"grid_size": grid_size, "mask_area_ratio": 0.4, "mask_ar_range": (0.75, 1.5), "num_sub_masks": 2},
        {"grid_size": grid_size, "mask_area_ratio": 0.15, "mask_ar_range": (0.75, 1.5), "num_sub_masks": 1},
    ]
    collator = MaskCollator(cfgs)

    B, T, C, H, W = 4, 16, 1, 160, 144
    batch = [
        {
            "video": torch.randn(T, C, H, W),
            "clip_number": i,
            "start_frame": 0,
            "end_frame": T - 1,
            "source_video": "x",
        }
        for i in range(B)
    ]
    out = collator(batch)

    # video is stacked normally
    assert out["video"].shape == (B, T, C, H, W)

    # one mask pair per config, each shaped [B, K]
    assert len(out["masks_enc"]) == len(cfgs)
    assert len(out["masks_pred"]) == len(cfgs)
    for me, mp in zip(out["masks_enc"], out["masks_pred"]):
        assert me.shape[0] == B and mp.shape[0] == B


def test_mask_collator_passes_through_dataset_keys():
    """Non-video dataset fields (`clip_number`, `start_frame`, `source_video`,
    etc.) survive collation untouched, so loggers and debug tooling can still
    trace a batch back to its source clips."""
    grid_size = (8, 10, 9)
    cfgs = [{"grid_size": grid_size, "mask_area_ratio": 0.4, "mask_ar_range": (0.75, 1.5), "num_sub_masks": 1}]
    collator = MaskCollator(cfgs)

    batch = [
        {
            "video": torch.randn(16, 1, 160, 144),
            "clip_number": i,
            "start_frame": i * 16,
            "end_frame": i * 16 + 15,
            "source_video": "x",
        }
        for i in range(3)
    ]
    out = collator(batch)

    # non-tensor keys are stacked into batched tensors by default_collate
    assert torch.equal(out["clip_number"], torch.tensor([0, 1, 2]))
    assert torch.equal(out["start_frame"], torch.tensor([0, 16, 32]))
    assert out["source_video"] == ["x", "x", "x"]

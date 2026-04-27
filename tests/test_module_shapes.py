import torch
from agent_1.models.utils.modules import (
    PatchEmbed,
    RoPE,
    Attention,
    GatedMLP,
    TransformerBlock,
    ViT,
    apply_masks,
    grid_positions,
)


def test_patch_embed_2d():
    B, C, H, W = 2, 3, 16, 16
    patch_dim = (4, 4)
    embed_dim = 8

    # num_patches = (H/pH) * (W/pW) = 4 * 4 = 16
    num_patches = (H // patch_dim[0]) * (W // patch_dim[1])

    model = PatchEmbed(patch_dim=patch_dim, in_channels=C, embed_dim=embed_dim)
    x = torch.randn(B, C, H, W)
    out = model(x)

    # (B, num_patches, embed_dim) = (2, 16, 8)
    assert out.shape == (B, num_patches, embed_dim), f"expected {(B, num_patches, embed_dim)}, got {out.shape}"


def test_patch_embed_3d():
    B, C, T, H, W = 2, 3, 4, 16, 16
    patch_dim = (2, 4, 4)
    embed_dim = 8

    # num_patches = (T/pT) * (H/pH) * (W/pW) = 2 * 4 * 4 = 32
    num_patches = (T // patch_dim[0]) * (H // patch_dim[1]) * (W // patch_dim[2])

    model = PatchEmbed(patch_dim=patch_dim, in_channels=C, embed_dim=embed_dim)
    x = torch.randn(B, C, T, H, W)
    out = model(x)

    # (B, num_patches, embed_dim) = (2, 32, 8)
    assert out.shape == (B, num_patches, embed_dim), f"expected {(B, num_patches, embed_dim)}, got {out.shape}"


def test_rope_1d():
    B, heads, head_dim = 2, 4, 8
    grid_size = (8,)
    dim_partitions = (8,)

    # 1D sequence: N = 8 positions
    N = grid_size[0]

    rope = RoPE(head_dim=head_dim, grid_size=grid_size, dim_partitions=dim_partitions)
    q = torch.randn(B, heads, N, head_dim)
    k = torch.randn(B, heads, N, head_dim)
    q_out, k_out = rope(q, k)

    # output shapes match input: (B, heads, N, head_dim) = (2, 4, 8, 8)
    assert q_out.shape == (B, heads, N, head_dim), f"expected {(B, heads, N, head_dim)}, got {q_out.shape}"
    assert k_out.shape == (B, heads, N, head_dim), f"expected {(B, heads, N, head_dim)}, got {k_out.shape}"


def test_rope_2d():
    B, heads, head_dim = 2, 4, 8
    grid_size = (4, 4)
    dim_partitions = (4, 4)

    # 2D grid: N = 4 * 4 = 16 positions
    N = grid_size[0] * grid_size[1]

    rope = RoPE(head_dim=head_dim, grid_size=grid_size, dim_partitions=dim_partitions)
    q = torch.randn(B, heads, N, head_dim)
    k = torch.randn(B, heads, N, head_dim)
    q_out, k_out = rope(q, k)

    # output shapes match input: (B, heads, N, head_dim) = (2, 4, 16, 8)
    assert q_out.shape == (B, heads, N, head_dim), f"expected {(B, heads, N, head_dim)}, got {q_out.shape}"
    assert k_out.shape == (B, heads, N, head_dim), f"expected {(B, heads, N, head_dim)}, got {k_out.shape}"


def test_rope_3d():
    B, heads, head_dim = 2, 4, 12
    grid_size = (2, 4, 4)
    dim_partitions = (4, 4, 4)

    # 3D grid: N = 2 * 4 * 4 = 32 positions
    N = grid_size[0] * grid_size[1] * grid_size[2]

    rope = RoPE(head_dim=head_dim, grid_size=grid_size, dim_partitions=dim_partitions)
    q = torch.randn(B, heads, N, head_dim)
    k = torch.randn(B, heads, N, head_dim)
    q_out, k_out = rope(q, k)

    # output shapes match input: (B, heads, N, head_dim) = (2, 4, 32, 12)
    assert q_out.shape == (B, heads, N, head_dim), f"expected {(B, heads, N, head_dim)}, got {q_out.shape}"
    assert k_out.shape == (B, heads, N, head_dim), f"expected {(B, heads, N, head_dim)}, got {k_out.shape}"


def test_rope_partial_sequence():
    B, heads, head_dim = 2, 4, 8
    grid_size = (4, 4)
    dim_partitions = (4, 4)

    # grid has 16 total positions, but we only use the first 10
    # forward slices cos/sin to q.shape[-2], so N < total grid size works
    N = 10

    rope = RoPE(head_dim=head_dim, grid_size=grid_size, dim_partitions=dim_partitions)
    q = torch.randn(B, heads, N, head_dim)
    k = torch.randn(B, heads, N, head_dim)
    q_out, k_out = rope(q, k)

    # output shapes match input: (B, heads, N, head_dim) = (2, 4, 10, 8)
    assert q_out.shape == (B, heads, N, head_dim), f"expected {(B, heads, N, head_dim)}, got {q_out.shape}"
    assert k_out.shape == (B, heads, N, head_dim), f"expected {(B, heads, N, head_dim)}, got {k_out.shape}"


def test_attention():
    B, N, num_heads, head_dim = 2, 16, 4, 8
    embed_dim = num_heads * head_dim  # 4 * 8 = 32

    model = Attention(num_heads=num_heads, head_dim=head_dim)
    x = torch.randn(B, N, embed_dim)
    out = model(x)

    # (B, N, embed_dim) = (2, 16, 32)
    assert out.shape == (B, N, embed_dim), f"expected {(B, N, embed_dim)}, got {out.shape}"


def test_attention_with_rope():
    B, N, num_heads, head_dim = 2, 16, 4, 8
    embed_dim = num_heads * head_dim  # 4 * 8 = 32

    # RoPE for a 4x4 grid matching N=16
    rope = RoPE(head_dim=head_dim, grid_size=(4, 4), dim_partitions=(4, 4))

    model = Attention(num_heads=num_heads, head_dim=head_dim, rope=rope)
    x = torch.randn(B, N, embed_dim)
    out = model(x)

    # (B, N, embed_dim) = (2, 16, 32)
    assert out.shape == (B, N, embed_dim), f"expected {(B, N, embed_dim)}, got {out.shape}"


def test_gated_mlp():
    B, N = 2, 16
    hidden_size, intermediate_size = 32, 64

    model = GatedMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
    x = torch.randn(B, N, hidden_size)
    out = model(x)

    # (B, N, hidden_size) = (2, 16, 32)
    assert out.shape == (B, N, hidden_size), f"expected {(B, N, hidden_size)}, got {out.shape}"


def test_transformer_block():
    B, N = 2, 16
    hidden_size, intermediate_size = 32, 64
    num_heads, head_dim = 4, 8

    model = TransformerBlock(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    x = torch.randn(B, N, hidden_size)
    out = model(x)

    # (B, N, hidden_size) = (2, 16, 32)
    assert out.shape == (B, N, hidden_size), f"expected {(B, N, hidden_size)}, got {out.shape}"


def test_vit_2d():
    B, C, H, W = 2, 3, 16, 16
    patch_dim = (4, 4)
    hidden_size, intermediate_size = 32, 64
    num_heads, head_dim = 4, 8
    num_layers = 2

    grid_size = (H // patch_dim[0], W // patch_dim[1])
    num_patches = grid_size[0] * grid_size[1]

    model = ViT(
        patch_dim=patch_dim,
        in_channels=C,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        grid_size=grid_size,
        dim_partitions=(4, 4),
    )

    x = torch.randn(B, C, H, W)
    out = model(x)

    assert out.shape == (B, num_patches, hidden_size), f"expected {(B, num_patches, hidden_size)}, got {out.shape}"


def test_vit_3d():
    B, C, T, H, W = 2, 3, 4, 16, 16
    patch_dim = (2, 4, 4)
    hidden_size, intermediate_size = 32, 64
    num_heads, head_dim = 4, 8
    num_layers = 2

    grid_size = (T // patch_dim[0], H // patch_dim[1], W // patch_dim[2])
    num_patches = grid_size[0] * grid_size[1] * grid_size[2]
    dim_partitions = (2, 4, 2)

    model = ViT(
        patch_dim=patch_dim,
        in_channels=C,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        grid_size=grid_size,
        dim_partitions=dim_partitions,
    )

    x = torch.randn(B, C, T, H, W)
    out = model(x)

    assert out.shape == (B, num_patches, hidden_size), f"expected {(B, num_patches, hidden_size)}, got {out.shape}"


def test_apply_masks_gathers_per_sample():
    """`apply_masks` selects K out of N tokens per sample, preserving the
    feature dim — this is the gather contract every masked path depends on."""
    B, N, D = 3, 16, 8
    K = 5

    x = torch.randn(B, N, D)
    indices = torch.stack([torch.randperm(N)[:K] for _ in range(B)])
    out = apply_masks(x, indices)

    assert out.shape == (B, K, D)
    # spot-check one sample: the selected rows match
    for b in range(B):
        for j in range(K):
            assert torch.equal(out[b, j], x[b, indices[b, j]])


def test_grid_positions_roundtrip_3d():
    """`grid_positions(arange(T*H*W), (T, H, W))` reproduces the row-major
    meshgrid — this is the inverse of flattening, which RoPE relies on to
    rotate by original-grid coordinates after gathering."""
    T, H, W = 3, 4, 5
    flat = torch.arange(T * H * W)

    t_pos, h_pos, w_pos = grid_positions(flat, (T, H, W))

    expected_t, expected_h, expected_w = torch.meshgrid(
        torch.arange(T), torch.arange(H), torch.arange(W), indexing="ij"
    )
    assert torch.equal(t_pos, expected_t.flatten())
    assert torch.equal(h_pos, expected_h.flatten())
    assert torch.equal(w_pos, expected_w.flatten())


def test_rope_position_invariance_under_masking():
    """RoPE applied to a token at original position p via per-axis position
    indices yields the same rotation as picking out that token from the
    full-grid result — the V-JEPA encoder relies on this so masked context
    tokens carry correct positional phase."""
    B, heads, head_dim = 1, 2, 8
    grid_size = (4, 4)
    dim_partitions = (4, 4)
    N = grid_size[0] * grid_size[1]

    rope = RoPE(head_dim=head_dim, grid_size=grid_size, dim_partitions=dim_partitions)
    q = torch.randn(B, heads, N, head_dim)
    k = torch.randn(B, heads, N, head_dim)
    q_full, k_full = rope(q, k)  # default sequential positions

    # pick a scattered subset and apply RoPE with explicit positions
    pick = torch.tensor([[0, 5, 11, 14]])  # [B=1, K=4]
    q_sub = q[:, :, pick[0], :]
    k_sub = k[:, :, pick[0], :]
    positions = grid_positions(pick, grid_size)
    q_rot, k_rot = rope(q_sub, k_sub, positions)

    assert torch.allclose(q_rot, q_full[:, :, pick[0], :], atol=1e-5)
    assert torch.allclose(k_rot, k_full[:, :, pick[0], :], atol=1e-5)


def test_vit_with_indices():
    """Passing `indices` returns one embedding per kept token — the encoder
    sees only the visible subset, not the full grid."""
    B, C, T, H, W = 2, 3, 4, 16, 16
    patch_dim = (2, 4, 4)
    hidden_size, intermediate_size = 32, 64
    num_heads, head_dim = 4, 8
    num_layers = 2

    grid_size = (T // patch_dim[0], H // patch_dim[1], W // patch_dim[2])
    num_patches = grid_size[0] * grid_size[1] * grid_size[2]

    model = ViT(
        patch_dim=patch_dim,
        in_channels=C,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        grid_size=grid_size,
        dim_partitions=(2, 4, 2),
    )

    x = torch.randn(B, C, T, H, W)
    K = 12
    indices = torch.stack([torch.randperm(num_patches)[:K] for _ in range(B)])
    out = model(x, indices)

    assert out.shape == (B, K, hidden_size)

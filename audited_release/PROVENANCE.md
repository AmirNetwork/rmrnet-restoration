# Result provenance

## Reported controlled model

The implementation under `reported_controlled_main/` is the archived source compatible with the two 30-epoch controlled-training checkpoints used in the manuscript.

| Artifact | SHA-256 |
|---|---|
| `reported_controlled_main/rcadnet/model.py` | `F650AC33258DF76DFC954DA8D112A3B6CD2C1D3D154A25AD74FE0B53BA3934BB` |
| `reported_controlled_main/rcadnet/task_losses.py` | `F393798FCAD414F01218E27A76AC5650D917114A8B2F2CDB1A7311407683C0FC` |
| `reported_controlled_main/train_rcadnet.py` | `B2A9B9F216D2D576B916EC820C62C1047D4440BDA837005D50452502321F1E27` |
| `reported_checkpoints/ivcnz_epoch_028.pth` | `3C6A1A8E582639FADE7C5CAE9CBB301E3F2987D600C06584ADACB14C1AB538DD` |
| `reported_checkpoints/pcm_epoch_028.pth` | `E580D2BF0BB8CC3319AFBCCA3B3D1CB96EF340667497F445B6685C2EBBE7EEC7` |

The three source hashes above are the pre-packaging audit hashes recorded with the checkpoints. The release adds comments and author/provenance headers only, so current source-file hashes differ while Python behavior and state-dictionary structure remain unchanged. `FILE_HASHES_SHA256.txt` records the files actually shipped, and `verify_reported_checkpoints.py` is the executable compatibility check.

Both checkpoints contain 223 model tensors. `verify_reported_checkpoints.py` reconstructs the architecture from the recorded configuration, performs a strict load, and runs a tensor-shape inference check.

## Selection and leakage safeguards

- Each epoch was saved.
- Epoch 28 was selected by mean frozen-detector validation mAP50 across the four controlled degradation families.
- The test split was evaluated after selection and did not determine the checkpoint.
- The selection summaries and best-checkpoint records are in `reported_results/`.
- The manuscript discloses retrospective duplicate and temporal-adjacency findings in the legacy IVCNZ and PCM image-level partitions and limits claims accordingly.

## Later variants

`variants/current_research/` records later architectural work used for subsequent field-policy research. It must not be substituted silently when reproducing the controlled tables. Any result produced with that code should carry its own configuration, checkpoint hash, and selection ledger.

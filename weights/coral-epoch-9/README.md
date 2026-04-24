# CORAL trained weights (epoch 9)

This directory is the release target for the final CORAL checkpoint used in
the paper. The weights are hosted on Zenodo rather than checked into git
because the LoRA safetensors file exceeds GitHub's 100 MB per-file limit.

To populate this directory, run:

```bash
bash scripts/download_weights.sh
```

The script will download and extract:

- `adapter_config.json`  LoRA + `modules_to_save` configuration.
- `adapter_model.safetensors`  trained LoRA adapters and the cross-modality
  encoder / pooler / classifier weights saved via `modules_to_save`.

Once extracted, load with:

```python
from peft import PeftModel
from coral.model import CoralConfig, CoralPretraining

config = CoralConfig(
    rna_encoder_checkpoint="./DNABERT-2-117M",
    protein_encoder_checkpoint="facebook/esm2_t30_150M_UR50D",
)
base = CoralPretraining(config=config)
model = PeftModel.from_pretrained(base, "weights/coral-epoch-9").merge_and_unload()
```

See the main `README.md` for end-to-end usage and reproduction commands.

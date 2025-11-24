# Model Path Verification

This document verifies that model paths generated during pretraining match the paths expected during testing/validation.

## ASAP7 PDK

### MLP Model Paths

**Pretraining** (`ASAP7_MLP_voltage_variation_pretraining.py:241`):
```
pretrained_asap7_{cell_type}_{data_type}_{corner}_test5_{model_type}_{num_iterations}.pth
```

**Validation** (`ASAP7_voltage_variation_validation.py:141`):
```
pretrained_asap7_{cell_type}_{data_type}_{corner}_test5_{model_type}_{args.num_iterations}.pth
```

**Status:** ✅ **MATCH**

**Example:**
- Training: `pretrained_asap7_lvt_cell_FF_test5_aadam_100000.pth`
- Testing: `pretrained_asap7_lvt_cell_FF_test5_aadam_100000.pth`

---

### MAML Model Paths

**Pretraining** (`ASAP7_MAML_voltage_variation_pretraining.py:298`):
```
{data_type}_innerdiv{innerdiv}_meta{meta}_full1DMAML_weights_3hidden_({layer_length})_{total_iterations}_{cell_type.upper()}_{corner}_test5(dim5)_inner{inner_step}_fixed.pth
```

**Validation** (`ASAP7_voltage_variation_validation.py:143`):
```
{data_type}_innerdiv{args.innerdiv}_meta{args.meta}_full1DMAML_weights_3hidden_({args.layer_length})_{args.num_iterations}_{cell_type.upper()}_{corner}_test5(dim5)_inner{args.inner_step}_fixed.pth
```

**Status:** ✅ **MATCH**

**Example:**
- Training: `cell_innerdiv10_meta16_full1DMAML_weights_3hidden_(40)_100000_LVT_FF_test5(dim5)_inner3_fixed.pth`
- Testing: `cell_innerdiv10_meta16_full1DMAML_weights_3hidden_(40)_100000_LVT_FF_test5(dim5)_inner3_fixed.pth`

---

## TSMC PDK

### MLP Model Paths

**Pretraining** (`TSMC_MLP_voltage_variation_pretraining.py:121`):
```
pretrained_tsmc_{corner.upper()}_{temp}_test5_{data_type}_{model_type}_{num_iterations}.pth
```

**Validation** (`TSMC_voltage_variation_validation.py:147`):
```
pretrained_tsmc_{corner.upper()}_{temp}_test5_{data_type}_{model_type}_{args.num_iterations}.pth
```

**Status:** ✅ **MATCH**

**Example:**
- Training: `pretrained_tsmc_FF_25_test5_cell_aadam_30000.pth`
- Testing: `pretrained_tsmc_FF_25_test5_cell_aadam_30000.pth`

---

### MAML Model Paths

**Pretraining** (`TSMC_MAML_voltage_variation_pretraining.py:329`):
```
{data_type}_innerdiv{innerdiv}_meta{meta}_full1DMAML_weights_3hidden_({layer_length})_{total_iterations}_TSMC_{condition_type.upper()}_{temp}_test5(dim5)_inner{inner_step}.pth
```

**Validation** (`TSMC_voltage_variation_validation.py:149`):
```
{data_type}_innerdiv{args.innerdiv}_meta{args.meta}_full1DMAML_weights_3hidden_({args.layer_length})_{args.num_iterations}_TSMC_{corner.upper()}_{temp}_test5(dim5)_inner{args.inner_step}.pth
```

**Status:** ✅ **MATCH**

**Example:**
- Training: `cell_innerdiv10_meta32_full1DMAML_weights_3hidden_(40)_30000_TSMC_FF_25_test5(dim5)_inner1.pth`
- Testing: `cell_innerdiv10_meta32_full1DMAML_weights_3hidden_(40)_30000_TSMC_FF_25_test5(dim5)_inner1.pth`

---

## Directory Structure

### Pretraining Output Directories

**MLP Models:**
```
model_pretraining_code/
└── MLP_pretrained_model/
    ├── pretrained_asap7_*.pth    # ASAP7 MLP models
    └── pretrained_tsmc_*.pth     # TSMC MLP models
```

**MAML Models:**
```
pretrained_models/
└── taskdivide_all/
    ├── cell_innerdiv*_ASAP7_*.pth      # ASAP7 MAML models
    ├── transition_innerdiv*_ASAP7_*.pth
    ├── cell_innerdiv*_TSMC_*.pth       # TSMC MAML models
    └── transition_innerdiv*_TSMC_*.pth
```

**MAML Checkpoints:**
```
pretrained_models/
└── checkpoints/
    └── taskdivide_all_checkpoints/
        ├── *_ASAP7_*_checkpoint_*.pth
        └── *_TSMC_*_checkpoint_*.pth
```

---

## Key Parameter Mappings

### ASAP7

| Parameter | MLP Default | MAML Default | Notes |
|-----------|-------------|--------------|-------|
| `inner_step` | N/A | 3 | MAML inner loop steps |
| `innerdiv` | N/A | 10 | Inner LR = 0.001/innerdiv |
| `meta` | N/A | 16 | Tasks per meta batch |
| `layer_length` | N/A | 40 | Hidden layer size |
| `model_type` | aadam | N/A | aadam=256, mlp=40 hidden |
| `num_iterations` | 100000 | 100000 | Training iterations |

### TSMC

| Parameter | MLP Default | MAML Default | Notes |
|-----------|-------------|--------------|-------|
| `inner_step` | N/A | 1 | MAML inner loop steps |
| `innerdiv` | N/A | 10 | Inner LR = 0.001/innerdiv |
| `meta` | N/A | 32 | Tasks per meta batch |
| `layer_length` | N/A | 40 | Hidden layer size |
| `model_type` | aadam | N/A | aadam=256, mlp=40 hidden |
| `num_iterations` | 30000 | 30000 | Training iterations |

---

## Verification Checklist

- [x] ASAP7 MLP paths match between training and testing
- [x] ASAP7 MAML paths match between training and testing
- [x] TSMC MLP paths match between training and testing
- [x] TSMC MAML paths match between training and testing
- [x] All parameter defaults are consistent
- [x] Directory structures are documented
- [x] Checkpoint paths are consistent

---

## Testing Path Match

To verify a model can be loaded:

```bash
# 1. Train a model
python run_voltage_variation_pretraining.py --pdk asap7 --model mlp \
    --corner FF --cell_type lvt --num_iterations 100000

# 2. Test the same model
python ../model_test_code/run_voltage_variation_validation.py --pdk asap7 --model mlp \
    --corner FF --cell_type lvt --num_iterations 100000
```

The validation script should automatically find and load the pretrained model with matching parameters.

---

## Notes

1. **Case Sensitivity**:
   - ASAP7 MAML uses `{cell_type.upper()}` → LVT, RVT, SLVT, SRAM
   - TSMC uses `{corner.upper()}` → FF, SS, TT

2. **Fixed Suffix**:
   - ASAP7 MAML models end with `_fixed.pth`
   - TSMC MAML models do NOT have `_fixed` suffix

3. **Test5(dim5) Indicator**:
   - All models include `test5` or `test5(dim5)` indicating 5-dimensional input

4. **Temperature Format**:
   - TSMC includes temperature as integer: `_25_`, `_100_`
   - No temperature in ASAP7 paths

---

**Last Updated:** 2024-11-24
**Status:** All paths verified ✅

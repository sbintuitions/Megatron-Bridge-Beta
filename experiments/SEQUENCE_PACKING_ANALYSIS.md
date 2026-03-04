# Sequence Packing Analysis for Qwen3-VL finetuning

## Where is Sequence Packing Performed?

### Two Different Implementations:

**1. In `qwen3_vl_step.py` (Qwen3-VL specific):**
- **File**: [src/megatron/bridge/models/qwen_vl/qwen3_vl_step.py](src/megatron/bridge/models/qwen_vl/qwen3_vl_step.py#L154)
- **Function**: `pack_or_pad_batch_sequences()` (lines 154-211)
- **What it does**: 
  - Creates packed sequence metadata (cu_seqlens, max_seqlen) for THD (Tensor Head Dimension) format
  - Does NOT concatenate sequences - sequences remain separate in the batch
  - Modifies attention masks and loss masks to work with packed sequences
  - Returns parameters for the Qwen3-VL model's forward pass

**2. In `vlm_step.py` (Generic VLM):**
- **File**: [src/megatron/bridge/training/vlm_step.py](src/megatron/bridge/training/vlm_step.py#L100)
- **Function**: `pack_batch_sequences()` (lines 100-225)
- **What it does**:
  - **Concatenates sequences** from a batch into a single long sequence
  - Example: [batch_size=2, seq_len=8] → [batch_size=1, seq_len=concatenated]
  - Removes padding tokens and recalculates position IDs
  - Produces cumulative sequence lengths (cu_seqlens) for attention masking per-sequence
  - Called in `get_batch()` function (line 308)

---

## How Sequence Packing Works in sft.sh Training

### Configuration Flag
```bash
dataset.pack_sequences_in_batch=$pack_config  # True or False
```

### The Flow:

1. **Dataset Creation** (MockVLMConversationProvider or similar):
   - When `pack_sequences_in_batch=True`: Creates multiple examples with varied response lengths
   - When `pack_sequences_in_batch=False`: Creates single minimal example that repeats

2. **Collate Function** (in [vlm_datasets/collate.py](src/megatron/bridge/data/vlm_datasets/collate.py)):
   - Processes conversation examples individually
   - No concatenation happens here - just applies processor and creates attention masks
   - Returns normal batched tensors with shape [batch_size, seq_len]

3. **Packing in Training Loop** (in [training/vlm_step.py](src/megatron/bridge/training/vlm_step.py#L308)):
   - `get_batch()` checks `enable_packing = getattr(cfg.dataset, "pack_sequences_in_batch", False)`
   - If `True`: Calls `pack_batch_sequences()` which concatenates batch items
   - If `False`: Pads to fixed length normally

4. **Forward Step** (in [models/qwen_vl/qwen3_vl_step.py](src/megatron/bridge/models/qwen_vl/qwen3_vl_step.py#L251)):
   - Receives already-paced sequences from dataset
   - Calls `pack_or_pad_batch_sequences()` to create model-specific parameters
   - Reshapes loss_mask and labels for Qwen3-VL's THD format
   - Passes packed_seq_params to model for attention computation

---

## Key Sequence Packing Implementation Details

### pack_batch_sequences() Logic:
```
Input:  tokens shape [batch_size=2, seq_len=8]
        Seq 1: [1, 2, 3, 0, 0, 0, 0, 0]  (actual_len=3, padded=8)
        Seq 2: [4, 5, 6, 7, 8, 0, 0, 0]  (actual_len=5, padded=8)

Process:
  1. Strip padding: seq_len_1 = 3, seq_len_2 = 5
  2. Concatenate: [1, 2, 3] + [4, 5, 6, 7, 8] = [1, 2, 3, 4, 5, 6, 7, 8]
  3. Calculate cu_seqlens = [0, 3, 8]  (cumulative lengths)
  4. Recalculate position_ids per-sequence: [[0,1,2], [0,1,2,3,4]]

Output: tokens shape [1, 8]
        cu_seqlens [0, 3, 8]  - used for attention masking
        max_seqlen 5
```

### Impact on Training:
- **With packing**: Better GPU memory utilization, reduced padding waste
- **Without packing**: Fixed-length sequences, simpler model logic
- **Requirement**: `micro_batch_size > 1` when using packing (prevents loss computation issues)

---

## For Different Data Types (Text-Only or Interleaved)

### Do you need to re-implement packing?

**Answer: NO - The packing implementation is DATA-AGNOSTIC.**

The key insight: Packing operates **after** the collate function, at the tensor level:

1. **Collate function** (in collate.py):
   - Handles all modality-specific processing (images, audio, text)
   - Converts to input_ids, labels, loss_mask tensors
   - Returns standard PyTorch tensors

2. **Packing layer** (pack_batch_sequences):
   - Only sees tensors, doesn't know what modalities created them
   - Works with ANY data type as long as they're converted to ID sequences

### To use different data types, you need to:

1. **Create new dataset provider** that returns examples in the correct format:
   - Can use existing providers as templates (mock_provider.py, hf_provider.py, preloaded_provider.py)
   - Implement `build_datasets()` to return VLMConversationDataset

2. **Create new collate function** (optional):
   - Only if your data type requires special modality handling
   - Add entry to `COLLATE_FNS` dictionary if needed
   - Otherwise, use existing collate_fn

3. **Set packing flag in script**:
   ```bash
   dataset.pack_sequences_in_batch=True
   ```
   - Same flag works for text-only, interleaved data, etc.

4. **NO changes needed to**:
   - `pack_batch_sequences()` function
   - `qwen3_vl_step.py` packing code
   - Training loop logic

### Example for Text-Only Training:
```python
# In a new text_only_provider.py
class TextOnlyDatasetProvider(BaseProvider):
    pack_sequences_in_batch: bool = False
    
    def build_datasets(self, context):
        # Just return text examples as {"conversation": [...]}
        # Don't include images
        dataset = VLMConversationDataset(
            base_examples=examples,
            target_length=samples,
            processor=self.processor,
        )
        return dataset, val_ds, test_ds

# In sft.sh:
dataset.maker_name=make_text_only_dataset
dataset.pack_sequences_in_batch=True  # Same flag works!
```

---

## Summary

- **Packing happens at TWO levels**:
  1. Dataset level: Varies sequence lengths within batch (controlled by provider)
  2. Training loop level: Concatenates variable-length sequences (pack_batch_sequences)

- **Packing is DATA-AGNOSTIC**: Works with any modality (text, image, audio)

- **For different data types**: Only create new dataset provider/collate fn, reuse packing logic

- **No re-implementation needed**: Same `pack_batch_sequences()` works for all data types

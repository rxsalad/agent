import torch

from transformers import AutoProcessor, AutoModelForMultimodalLM


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------
model_name = "Qwen/Qwen3.6-27B"


# ---------------------------------------------------------
# Load processor
# ---------------------------------------------------------
print(f"Loading processor: {model_name}")

processor = AutoProcessor.from_pretrained(model_name)


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
print(f"Loading model: {model_name}")

model = AutoModelForMultimodalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
)

print("\nModel loaded successfully.")


# ---------------------------------------------------------
# 1. Model class
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("MODEL CLASS")
print("=" * 80)

print(model.__class__)


# ---------------------------------------------------------
# 2. Model configuration
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("MODEL CONFIGURATION")
print("=" * 80)

print(model.config)


# ---------------------------------------------------------
# 3. Important architecture parameters
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("ARCHITECTURE PARAMETERS")
print("=" * 80)

config = model.config

attributes = [
    "model_type",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "max_position_embeddings",
    "torch_dtype",
]

for attr in attributes:
    value = getattr(config, attr, "N/A")
    print(f"{attr:30s}: {value}")


# ---------------------------------------------------------
# 4. Entire model architecture
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("ENTIRE MODEL ARCHITECTURE")
print("=" * 80)

print(model)


# ---------------------------------------------------------
# 5. Top-level modules
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("TOP-LEVEL MODULES")
print("=" * 80)

for name, module in model.named_children():
    print(f"{name:30s}: {module.__class__.__name__}")


# ---------------------------------------------------------
# 6. Transformer layers
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("TRANSFORMER LAYERS")
print("=" * 80)

# Qwen models normally expose the transformer layers as:
# model.model.layers

if hasattr(model, "model") and hasattr(model.model, "layers"):

    layers = model.model.layers

    print(f"Number of layers: {len(layers)}")

    for i, layer in enumerate(layers):
        print(
            f"Layer {i:3d}: "
            f"{layer.__class__.__name__}"
        )

else:
    print("Could not find model.model.layers")


# ---------------------------------------------------------
# 7. Inspect the first transformer layer
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("FIRST TRANSFORMER LAYER")
print("=" * 80)

if hasattr(model, "model") and hasattr(model.model, "layers"):
    print(model.model.layers[0])


# ---------------------------------------------------------
# 8. Inspect the last transformer layer
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("LAST TRANSFORMER LAYER")
print("=" * 80)

if hasattr(model, "model") and hasattr(model.model, "layers"):
    print(model.model.layers[-1])


# ---------------------------------------------------------
# 9. Device placement
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("DEVICE MAP")
print("=" * 80)

if hasattr(model, "hf_device_map"):
    for device_name, device in model.hf_device_map.items():
        print(f"{device_name:50s}: {device}")
else:
    print("hf_device_map is not available.")


# ---------------------------------------------------------
# 10. Parameter count
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("PARAMETER COUNT")
print("=" * 80)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


# ---------------------------------------------------------
# Done
# ---------------------------------------------------------
print("\n" + "=" * 80)
print("DONE")
print("=" * 80)


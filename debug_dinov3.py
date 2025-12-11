import torch

try:
    print("Loading dinov3_vits16 (pretrained=False)...")
    model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vits16', pretrained=False)
    print("Successfully loaded dinov3_vits16")
    # print(model) # Output might be too large
    
    # Check for blocks
    if hasattr(model, 'blocks'):
        print("Model has 'blocks'")
        print("Last block type:", type(model.blocks[-1]))
        print("Last block:", model.blocks[-1])
        if hasattr(model.blocks[-1], 'attn'):
             print("Last block has 'attn'")
             print("Attn type:", type(model.blocks[-1].attn))
             print("Attn:", model.blocks[-1].attn)
    else:
        print("Unknown structure")
        
except Exception as e:
    print(f"Failed to load/inspect: {e}")

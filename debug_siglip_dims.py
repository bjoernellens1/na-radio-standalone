
import torch
import open_clip

def test_siglip_dims():
    print("Loading SigLIP...")
    try:
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-16-SigLIP', pretrained='webli')
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Checking dimensions...")
    dummy_img = torch.randn(1, 3, 224, 224)
    dummy_text = open_clip.get_tokenizer('ViT-B-16-SigLIP')(["test"])

    with torch.no_grad():
        img_feat = model.encode_image(dummy_img)
        text_feat = model.encode_text(dummy_text)
    
    print(f"Image feature shape: {img_feat.shape}")
    print(f"Text feature shape: {text_feat.shape}")

    if img_feat.shape[-1] != text_feat.shape[-1]:
        print("MISMATCH DETECTED!")
    else:
        print("Dimensions match.")

if __name__ == "__main__":
    test_siglip_dims()

from acquisition.embed import load_model
from PIL import Image
import torch
from vilt.transforms import pixelbert_transform

def compute_itm(model_path, img_path, text):
    model, tokenizer = load_model(model_path)
    device = torch.device('cuda')
    model.to(device)

    image = Image.open(img_path)
    img = pixelbert_transform(size=384)(image)
    img = img.unsqueeze(0).to(device)

    batch = {"text": [text], "image": [img]}
    with torch.no_grad():
        encoded = tokenizer(batch["text"])
        batch["text_ids"] = torch.tensor(encoded["input_ids"]).to(device)
        batch["text_labels"] = torch.tensor(encoded["input_ids"]).to(device)
        batch["text_masks"] = torch.tensor(encoded["attention_mask"]).to(device)
        infer = model.infer(batch)
        itm_logits = model.itm_score(infer["cls_feats"])
        exp_logits = torch.exp(itm_logits)

    return (exp_logits[0, 1]/torch.sum(exp_logits)).item()

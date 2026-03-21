"""
Test CLIP for product classification:
1. CLIP alone — match crop embeddings to text descriptions of products
2. CLIP + OCR — combine visual similarity with OCR text for better matching

Usage:
    uv run python task1_object_detection/experiments/test_clip.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

TASK_ROOT = Path(__file__).resolve().parent.parent
CLASSIFIER_DIR = TASK_ROOT / "output" / "classifier_dataset" / "train"
ANNOTATIONS_FILE = TASK_ROOT / "data" / "coco_dataset" / "train" / "annotations.json"
OUTPUT_DIR = TASK_ROOT / "output" / "analysis"

# Confusable products grouped by super-category
TEST_GROUPS = {
    "knekkebroed": [
        (86, "HAVRE KNEKKEBRØD 300G WASA"),
        (109, "KNEKKEBRØD 100 FRØ&HAVSALT 245G WASA"),
        (349, "KNEKKEBRØD SPORT+ 210G WASA"),
        (246, "HUSMAN KNEKKEBRØD 260G WASA"),
        (132, "FRUKOST FULLKORN 320G WASA"),
        (271, "FRUKOST KNEKKEBRØD 240G WASA"),
    ],
    "coffee": [
        (100, "EVERGOOD CLASSIC FILTERMALT 250G"),
        (49, "EVERGOOD DARK ROAST FILTERMALT 250G"),
        (304, "EVERGOOD CLASSIC KOKMALT 250G"),
        (341, "EVERGOOD CLASSIC HELE BØNNER 500G"),
        (63, "ALI MØRKBRENT FILTERMALT 250G"),
        (171, "ALI ORIGINAL FILTERMALT 250G"),
    ],
    "tea": [
        (46, "GRØNN TE SITRON 25POS TWININGS"),
        (173, "EARL GREY TEA 25POS TWININGS"),
        (89, "FOREST FRUIT TEA 20POS LIPTON"),
        (189, "YELLOW LABEL TEA 25POS LIPTON"),
        (87, "SPICY INDIAN CHAI TEA 20POS TWININGS"),
    ],
    "cereal": [
        (218, "CORN FLAKES 500G KELLOGGS"),
        (97, "CHEERIOS MULTI 375G NESTLE"),
        (181, "CHEERIOS HAVRE 375G NESTLE"),
        (29, "NESQUIK FROKOSTBLANDING 375G NESTLE"),
    ],
}


def test():
    print("Loading CLIP model (openai/clip-vit-base-patch32)...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    print(f"Device: {device}")

    # Also try to load EasyOCR for combo test
    try:
        import easyocr
        ocr_reader = easyocr.Reader(["en", "no"], gpu=False, verbose=False)
        has_ocr = True
        print("EasyOCR loaded for combo test")
    except ImportError:
        has_ocr = False
        print("EasyOCR not available — testing CLIP only")

    results = {"clip_only": [], "clip_plus_ocr": []}

    for group_name, products in TEST_GROUPS.items():
        print(f"\n{'='*80}")
        print(f"GROUP: {group_name.upper()} ({len(products)} products)")
        print(f"{'='*80}")

        # Build text descriptions for all products in this group
        product_names = [name for _, name in products]
        text_descriptions = [f"a photo of {name}" for name in product_names]

        # Encode text descriptions
        text_inputs = processor(text=text_descriptions, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_out = model.get_text_features(**text_inputs)
            text_features = text_out if isinstance(text_out, torch.Tensor) else text_out.pooler_output
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Test each product
        for cat_id, expected_name in products:
            crop_dir = CLASSIFIER_DIR / str(cat_id)
            if not crop_dir.exists():
                continue

            crops = sorted(crop_dir.glob("*.jpg"))[:5]

            print(f"\n  [{cat_id}] {expected_name}")

            for crop_path in crops:
                img = Image.open(crop_path).convert("RGB")

                # CLIP image embedding
                image_inputs = processor(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    img_out = model.get_image_features(**image_inputs)
                    image_features = img_out if isinstance(img_out, torch.Tensor) else img_out.pooler_output
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Similarity to all product descriptions
                similarities = (image_features @ text_features.T).squeeze().cpu().numpy()
                pred_idx = np.argmax(similarities)
                pred_name = product_names[pred_idx]
                pred_score = similarities[pred_idx]

                correct = pred_name == expected_name
                expected_idx = product_names.index(expected_name)
                expected_score = similarities[expected_idx]

                # Rank of correct answer
                rank = int((similarities >= expected_score).sum())

                results["clip_only"].append({
                    "cat_id": cat_id,
                    "expected": expected_name,
                    "predicted": pred_name,
                    "correct": correct,
                    "rank": rank,
                    "pred_score": float(pred_score),
                    "expected_score": float(expected_score),
                    "group": group_name,
                })

                status = "✓" if correct else f"✗ (pred: {pred_name[:40]}, rank={rank})"
                print(f"    CLIP: {status} (score={expected_score:.3f})")

                # CLIP + OCR combo
                if has_ocr:
                    ocr_results = ocr_reader.readtext(str(crop_path))
                    ocr_text = " ".join(r[1] for r in ocr_results).upper()

                    # Re-rank using OCR text overlap
                    combo_scores = []
                    for i, name in enumerate(product_names):
                        clip_score = float(similarities[i])

                        # OCR bonus: how many words from product name appear in OCR text
                        name_words = name.upper().split()
                        ocr_matches = sum(1 for w in name_words if w in ocr_text)
                        ocr_bonus = ocr_matches / len(name_words) if name_words else 0

                        # Combine: CLIP similarity + OCR match bonus
                        combo = clip_score + 0.3 * ocr_bonus
                        combo_scores.append(combo)

                    combo_pred_idx = int(np.argmax(combo_scores))
                    combo_pred = product_names[combo_pred_idx]
                    combo_correct = combo_pred == expected_name

                    combo_rank = int(sum(1 for s in combo_scores if s >= combo_scores[expected_idx]))

                    results["clip_plus_ocr"].append({
                        "cat_id": cat_id,
                        "expected": expected_name,
                        "predicted": combo_pred,
                        "correct": combo_correct,
                        "rank": combo_rank,
                        "ocr_text": ocr_text[:100],
                        "group": group_name,
                    })

                    status2 = "✓" if combo_correct else f"✗ (pred: {combo_pred[:40]}, rank={combo_rank})"
                    print(f"    CLIP+OCR: {status2}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    for method in ["clip_only", "clip_plus_ocr"]:
        data = results[method]
        if not data:
            continue
        total = len(data)
        correct = sum(1 for r in data if r["correct"])
        top3 = sum(1 for r in data if r["rank"] <= 3)
        top5 = sum(1 for r in data if r["rank"] <= 5)

        print(f"\n  {method}:")
        print(f"    Top-1 accuracy: {correct}/{total} ({correct/total:.1%})")
        print(f"    Top-3 accuracy: {top3}/{total} ({top3/total:.1%})")
        print(f"    Top-5 accuracy: {top5}/{total} ({top5/total:.1%})")

        # Per-group breakdown
        for group in TEST_GROUPS:
            group_data = [r for r in data if r["group"] == group]
            if group_data:
                gc = sum(1 for r in group_data if r["correct"])
                print(f"    {group}: {gc}/{len(group_data)} ({gc/len(group_data):.0%})")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "clip_test_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {OUTPUT_DIR / 'clip_test_results.json'}")


if __name__ == "__main__":
    test()

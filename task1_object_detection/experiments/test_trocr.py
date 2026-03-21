"""
Test TrOCR (Transformer-based OCR) on product crops.
TrOCR is specifically designed for text recognition in images.

Usage:
    uv run python task1_object_detection/experiments/test_trocr.py
"""

import json
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

TASK_ROOT = Path(__file__).resolve().parent.parent
CLASSIFIER_DIR = TASK_ROOT / "output" / "classifier_dataset" / "train"

# Same confusable products as before
CONFUSABLE_PAIRS = [
    # WASA variants (hardest)
    (86, "HAVRE KNEKKEBRØD 300G WASA"),
    (109, "KNEKKEBRØD 100 FRØ&HAVSALT 245G WASA"),
    (349, "KNEKKEBRØD SPORT+ 210G WASA"),
    (246, "HUSMAN KNEKKEBRØD 260G WASA"),
    # EVERGOOD variants
    (100, "EVERGOOD CLASSIC FILTERMALT 250G"),
    (49, "EVERGOOD DARK ROAST FILTERMALT 250G"),
    (304, "EVERGOOD CLASSIC KOKMALT 250G"),
    # Tea (should be easier - different colors)
    (46, "GRØNN TE SITRON 25POS TWININGS"),
    (173, "EARL GREY TEA 25POS TWININGS"),
    (89, "FOREST FRUIT TEA 20POS LIPTON"),
    # Distinct products
    (218, "CORN FLAKES 500G KELLOGGS"),
    (97, "CHEERIOS MULTI 375G NESTLE"),
]


def test():
    print("Loading TrOCR model (microsoft/trocr-base-printed)...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    model.eval()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    print(f"Device: {device}")

    print(f"\n{'='*80}")
    print("TrOCR TEST ON PRODUCT CROPS")
    print(f"{'='*80}")

    results = []

    for cat_id, expected_name in CONFUSABLE_PAIRS:
        crop_dir = CLASSIFIER_DIR / str(cat_id)
        if not crop_dir.exists():
            print(f"\n  [{cat_id}] {expected_name}: NO CROPS")
            continue

        crops = sorted(crop_dir.glob("*.jpg"))[:3]

        print(f"\n  [{cat_id}] {expected_name}")

        for crop_path in crops:
            img = Image.open(crop_path).convert("RGB")
            w, h = img.size

            # TrOCR works best on text lines - try full crop and sub-regions
            texts = []

            # Full crop
            pixel_values = processor(img, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_new_tokens=50)
            full_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            texts.append(("full", full_text))

            # Top third (often has brand name)
            top_crop = img.crop((0, 0, w, h // 3))
            if top_crop.size[1] > 10:
                pv = processor(top_crop, return_tensors="pt").pixel_values.to(device)
                with torch.no_grad():
                    ids = model.generate(pv, max_new_tokens=30)
                texts.append(("top", processor.batch_decode(ids, skip_special_tokens=True)[0]))

            # Middle third
            mid_crop = img.crop((0, h // 3, w, 2 * h // 3))
            if mid_crop.size[1] > 10:
                pv = processor(mid_crop, return_tensors="pt").pixel_values.to(device)
                with torch.no_grad():
                    ids = model.generate(pv, max_new_tokens=30)
                texts.append(("mid", processor.batch_decode(ids, skip_special_tokens=True)[0]))

            # Bottom third (often has weight)
            bot_crop = img.crop((0, 2 * h // 3, w, h))
            if bot_crop.size[1] > 10:
                pv = processor(bot_crop, return_tensors="pt").pixel_values.to(device)
                with torch.no_grad():
                    ids = model.generate(pv, max_new_tokens=30)
                texts.append(("bot", processor.batch_decode(ids, skip_special_tokens=True)[0]))

            all_text = " ".join(t[1] for t in texts).upper()

            # Check word matches
            name_words = expected_name.upper().split()
            found_words = [w for w in name_words if w in all_text]
            match_ratio = len(found_words) / len(name_words) if name_words else 0

            print(f"    {crop_path.name} ({w}x{h}px)")
            for region, text in texts:
                print(f"      {region:4s}: {text[:80]}")
            print(f"      Matched: {found_words} ({match_ratio:.0%})")

            results.append({
                "cat_id": cat_id,
                "expected": expected_name,
                "crop": crop_path.name,
                "size": f"{w}x{h}",
                "texts": {r: t for r, t in texts},
                "all_text": all_text[:200],
                "match_ratio": match_ratio,
                "found_words": found_words,
            })

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    match_ratios = [r["match_ratio"] for r in results]
    if match_ratios:
        print(f"  Average word match ratio: {sum(match_ratios)/len(match_ratios):.2%}")
        print(f"  Crops with >50% match: {sum(1 for r in match_ratios if r > 0.5)}/{len(match_ratios)}")
        print(f"  Crops with >0% match: {sum(1 for r in match_ratios if r > 0)}/{len(match_ratios)}")
        print(f"  Crops with 0% match: {sum(1 for r in match_ratios if r == 0)}/{len(match_ratios)}")

    # Compare with EasyOCR results
    print(f"\n  (EasyOCR got 25% avg match for comparison)")

    # Save
    out_path = TASK_ROOT / "output" / "analysis" / "trocr_test_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    test()

"""
Test OCR on product crops to see if text can distinguish similar products.
Tests the theory that text (brand, weight, flavor) is the key signal.

Usage:
    uv run python task1_object_detection/experiments/test_ocr.py
"""

import json
from collections import defaultdict
from pathlib import Path

import easyocr
from PIL import Image

TASK_ROOT = Path(__file__).resolve().parent.parent
CLASSIFIER_DIR = TASK_ROOT / "output" / "classifier_dataset" / "train"
ANNOTATIONS_FILE = TASK_ROOT / "data" / "coco_dataset" / "train" / "annotations.json"

# Test on confusable pairs from the intra-class analysis
CONFUSABLE_PAIRS = [
    # Knekkebroed: WASA variants
    (86, "HAVRE KNEKKEBRØD 300G WASA"),
    (109, "KNEKKEBRØD 100 FRØ&HAVSALT 245G WASA"),
    (349, "KNEKKEBRØD SPORT+ 210G WASA"),
    (246, "HUSMAN KNEKKEBRØD 260G WASA"),
    (132, "FRUKOST FULLKORN 320G WASA"),
    # Coffee: EVERGOOD variants
    (100, "EVERGOOD CLASSIC FILTERMALT 250G"),
    (49, "EVERGOOD DARK ROAST FILTERMALT 250G"),
    (304, "EVERGOOD CLASSIC KOKMALT 250G"),
    (341, "EVERGOOD CLASSIC HELE BØNNER 500G"),
    # Tea: different flavors
    (46, "GRØNN TE SITRON 25POS TWININGS"),
    (173, "EARL GREY TEA 25POS TWININGS"),
    (89, "FOREST FRUIT TEA 20POS LIPTON"),
    # Easy to distinguish (for contrast)
    (218, "CORN FLAKES 500G KELLOGGS"),
    (97, "CHEERIOS MULTI 375G NESTLE"),
]


def test():
    print("Initializing EasyOCR (English + Norwegian)...")
    reader = easyocr.Reader(["en", "no"], gpu=False)

    with open(ANNOTATIONS_FILE) as f:
        coco = json.load(f)
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}

    print(f"\n{'='*80}")
    print("OCR TEST ON PRODUCT CROPS")
    print(f"{'='*80}")

    results = []

    for cat_id, expected_name in CONFUSABLE_PAIRS:
        crop_dir = CLASSIFIER_DIR / str(cat_id)
        if not crop_dir.exists():
            print(f"\n  [{cat_id}] {expected_name}: NO CROPS FOUND")
            continue

        crops = sorted(crop_dir.glob("*.jpg"))[:3]  # test 3 crops per product

        print(f"\n  [{cat_id}] {expected_name}")

        for crop_path in crops:
            img = Image.open(crop_path)
            w, h = img.size

            # Run OCR
            ocr_results = reader.readtext(str(crop_path))

            texts = [r[1] for r in ocr_results]
            confidences = [r[2] for r in ocr_results]
            all_text = " ".join(texts).upper()

            # Check if key distinguishing words are found
            name_words = expected_name.upper().split()
            found_words = [w for w in name_words if w in all_text]
            match_ratio = len(found_words) / len(name_words) if name_words else 0

            print(f"    {crop_path.name} ({w}x{h}px)")
            print(f"      OCR text: {all_text[:100]}")
            print(f"      Matched: {found_words} ({match_ratio:.0%})")
            if confidences:
                print(f"      Confidence: {min(confidences):.2f}-{max(confidences):.2f}")

            results.append({
                "cat_id": cat_id,
                "expected": expected_name,
                "crop": crop_path.name,
                "size": f"{w}x{h}",
                "ocr_text": all_text[:200],
                "match_ratio": match_ratio,
                "found_words": found_words,
                "num_detections": len(ocr_results),
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

    # Save results
    OUTPUT_DIR = TASK_ROOT / "output" / "analysis"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "ocr_test_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {OUTPUT_DIR / 'ocr_test_results.json'}")


if __name__ == "__main__":
    test()

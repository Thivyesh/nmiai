"""Fixed super-category mapping — used everywhere."""

SUPER_CATEGORIES = {
    "knekkebroed": ["knekkebrød", "knekke", "flatbrød", "wasa", "sigdal", "leksands", "ryvita", "korni"],
    "coffee": ["kaffe", "coffee", "espresso", "nescafe", "evergood", "friele", "dolce gusto", "cappuccino", "kapsel", "tassimo", "zoegas"],
    "tea": [" te ", "tea", "twinings", "lipton", "pukka", "urtete"],
    "cereal": ["frokost", "havre", "müsli", "granola", "corn flakes", "cheerios", "cruesli", "puffet", "fras", "weetabix", "fitness"],
    "eggs": ["egg"],
    "spread": ["smør", "bremykt", "brelett", "margarin", "olivero", "meierismør"],
    "cookies": ["kjeks", "cookie", "grissini", "surdeig"],
    "chocolate": ["sjokolade", "nugatti", "regia", "cocoa", "kakao"],
}

GROUP_NAMES = list(SUPER_CATEGORIES.keys()) + ["other"]


def get_group(cat_id, cat_map):
    name = cat_map.get(cat_id, "").lower()
    for g, kws in SUPER_CATEGORIES.items():
        if any(kw in name for kw in kws):
            return g
    return "other"

import os
import json
import argparse

# Paths
DATA_DIR = os.path.join(os.getcwd(), "data")

# --- OUTPUT FILES ---
CAPTIONS_OUT = os.path.join(DATA_DIR, "captions.json")
VQA_OUT = os.path.join(DATA_DIR, "vqa_train.json")
ANSWER_MAP_OUT = os.path.join(DATA_DIR, "answer_map.json")


def preprocess_captions(year="2014", max_samples=5000):
    """
    Convert COCO captions (2014 or 2017) into a flat JSON list:
      [{"image_id": "data/coco2014/train2014/000000xxxx.jpg", "caption": "..."}]
    """
    coco_dir = os.path.join(DATA_DIR, f"coco{year}")
    ann_path = os.path.join(coco_dir, "captions", f"captions_train{year}.json")

    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Captions annotation not found: {ann_path}")

    with open(ann_path, "r") as f:
        coco_data = json.load(f)

    results = []
    for ann in coco_data["annotations"][:max_samples]:
        img_file = os.path.join(coco_dir, "train" + year, f"{ann['image_id']:012d}.jpg")
        results.append({"image_id": img_file, "caption": ann["caption"]})

    with open(CAPTIONS_OUT, "w") as f:
        json.dump(results, f)

    print(f"✅ Saved {len(results)} caption samples from COCO {year} → {CAPTIONS_OUT}")


def preprocess_vqa(max_samples=5000):
    """
    Convert VQA annotations + questions into flat JSON:
      [{"image_id": "data/coco2014/train2014/000000xxxx.jpg", "question": "...", "answer": "..."}]
    Also saves an answer_map.json for classification.
    """
    vqa_dir = os.path.join(DATA_DIR, "vqa", "annotations")
    ann_path = os.path.join(vqa_dir, "v2_mscoco_train2014_annotations.json")
    ques_path = os.path.join(vqa_dir, "v2_OpenEnded_mscoco_train2014_questions.json")

    if not os.path.exists(ann_path) or not os.path.exists(ques_path):
        raise FileNotFoundError("VQA annotation/question files not found in vqa/annotations/")

    with open(ann_path, "r") as f:
        anns = json.load(f)
    with open(ques_path, "r") as f:
        ques = json.load(f)

    qa_pairs = []
    answer_map = {}
    ans_id = 0

    for q, a in zip(ques["questions"][:max_samples], anns["annotations"][:max_samples]):
        img_file = os.path.join(DATA_DIR, "coco2014", "train2014", f"{q['image_id']:012d}.jpg")
        answer = a["multiple_choice_answer"].lower()

        if answer not in answer_map:
            answer_map[answer] = ans_id
            ans_id += 1

        qa_pairs.append({"image_id": img_file, "question": q["question"], "answer": answer})

    with open(VQA_OUT, "w") as f:
        json.dump(qa_pairs, f)
    with open(ANSWER_MAP_OUT, "w") as f:
        json.dump(answer_map, f)

    print(f"✅ Saved {len(qa_pairs)} VQA samples → {VQA_OUT}")
    print(f"✅ Saved answer map with {len(answer_map)} unique answers → {ANSWER_MAP_OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=str, default="2014", choices=["2014", "2017"],
                        help="Choose COCO dataset year for captions preprocessing (2014 or 2017).")
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Limit number of samples for faster training/debugging.")
    args = parser.parse_args()

    # Run preprocessing
    preprocess_captions(year=args.year, max_samples=args.max_samples)
    preprocess_vqa(max_samples=args.max_samples)




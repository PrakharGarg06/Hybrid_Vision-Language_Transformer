import os
import json
import re
import subprocess
from threading import Thread
from flask import Flask, request, jsonify
from PIL import Image
import torch

from utils_hybrid import get_tokenizers_and_image_processor, load_checkpoint
from models_hybrid import VisionEncoderViT, CaptionDecoder, VQAHybrid

# ===============================================================
# Flask Setup
# ===============================================================

app = Flask(__name__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

t5_tokenizer, bert_tokenizer, image_processor = get_tokenizers_and_image_processor()

# ===============================================================
# Helper: Get Latest Checkpoint
# ===============================================================

def get_latest_checkpoint(folder, prefix):
    """Return latest checkpoint file by epoch number."""
    if not os.path.exists(folder):
        raise FileNotFoundError(f"❌ Folder not found: {folder}")
    files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(".pt")]
    if not files:
        raise FileNotFoundError(f"❌ No checkpoint found with prefix '{prefix}' in {folder}")
    files = sorted(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return os.path.join(folder, files[-1])

# ===============================================================
# Paths
# ===============================================================

CKPT_ROOT = r"C:\Users\Prakhar Garg\Desktop\multimodal-hybrid\checkpoints"
DATA_ROOT = r"C:\Users\Prakhar Garg\Desktop\multimodal-hybrid\data"

caption_dir = os.path.join(CKPT_ROOT, "caption")
vqa_original_dir = os.path.join(CKPT_ROOT, "vqa")
vqa_enriched_dir = os.path.join(CKPT_ROOT, "vqa_enriched")

CAPTION_CKPT = get_latest_checkpoint(caption_dir, "caption_epoch_")
VQA_CKPT_ORIG = get_latest_checkpoint(vqa_original_dir, "vqa_epoch_")
VQA_CKPT_ENRICHED = get_latest_checkpoint(vqa_enriched_dir, "vqa_epoch_")

print(f"✅ Using caption checkpoint: {CAPTION_CKPT}")
print(f"✅ Using VQA (original): {VQA_CKPT_ORIG}")
print(f"✅ Using VQA (enriched): {VQA_CKPT_ENRICHED}")

ANSWER_MAP_PATH_ORIG = os.path.join(DATA_ROOT, "answer_map.json")
ANSWER_MAP_PATH_ENRICHED = os.path.join(DATA_ROOT, "answer_map_enriched.json")

# Load answer maps
with open(ANSWER_MAP_PATH_ORIG, "r") as f:
    ans_map_orig = json.load(f)
with open(ANSWER_MAP_PATH_ENRICHED, "r") as f:
    ans_map_enriched = json.load(f)

idx2answer_orig = {v: k for k, v in ans_map_orig.items()}
idx2answer_enriched = {v: k for k, v in ans_map_enriched.items()}

# ===============================================================
# Load Models
# ===============================================================

vision_encoder = VisionEncoderViT(out_dim=768).to(DEVICE)
caption_decoder = CaptionDecoder(
    vocab_size=t5_tokenizer.vocab_size,
    d_model=768, nhead=8, num_layers=6,
    pad_idx=t5_tokenizer.pad_token_id
).to(DEVICE)

# Load shared caption checkpoint
print(f"✅ Loading caption checkpoint: {CAPTION_CKPT}")
ck_caption = load_checkpoint(CAPTION_CKPT, map_location=DEVICE)
vision_encoder.load_state_dict(ck_caption["vision"])
caption_decoder.load_state_dict(ck_caption["decoder"])

# Two VQA variants
vqa_model_orig = VQAHybrid(num_answers=len(ans_map_orig)).to(DEVICE)
vqa_model_enriched = VQAHybrid(num_answers=len(ans_map_enriched)).to(DEVICE)

print(f"✅ Loading VQA original checkpoint: {VQA_CKPT_ORIG}")
ck_vqa_orig = load_checkpoint(VQA_CKPT_ORIG, map_location=DEVICE)
vqa_model_orig.load_state_dict(ck_vqa_orig["model"])

print(f"✅ Loading VQA enriched checkpoint: {VQA_CKPT_ENRICHED}")
ck_vqa_enriched = load_checkpoint(VQA_CKPT_ENRICHED, map_location=DEVICE)
vqa_model_enriched.load_state_dict(ck_vqa_enriched["model"])

# ===============================================================
# Inference Functions
# ===============================================================

def generate_caption_from_image(pil_img, max_len=30):
    vision_encoder.eval()
    caption_decoder.eval()
    inputs = image_processor(images=pil_img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)
    with torch.no_grad():
        _, img_seq = vision_encoder(pixel_values)
        token_ids = caption_decoder.generate_greedy(img_seq, t5_tokenizer, max_len=max_len, device=DEVICE)
    return t5_tokenizer.decode(token_ids, skip_special_tokens=True)

def answer_question(pil_img, question, mode="original"):
    if mode == "enriched":
        vqa_model = vqa_model_enriched
        idx2answer = idx2answer_enriched
    else:
        vqa_model = vqa_model_orig
        idx2answer = idx2answer_orig

    vqa_model.eval()
    pixel_values = image_processor(images=pil_img, return_tensors="pt")["pixel_values"].to(DEVICE)
    q_enc = bert_tokenizer(question, return_tensors="pt", truncation=True, padding="max_length", max_length=32)
    input_ids, attention_mask = q_enc["input_ids"].to(DEVICE), q_enc["attention_mask"].to(DEVICE)
    with torch.no_grad():
        logits = vqa_model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
        pred = torch.argmax(logits, dim=-1).item()
    return idx2answer.get(pred, "unknown")

# ===============================================================
# Web UI
# ===============================================================

@app.route("/", methods=["GET"])
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Hybrid AI Demo</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-900 text-white flex items-center justify-center h-screen">
        <div class="bg-gray-800 p-8 rounded-2xl shadow-lg w-full max-w-xl text-center">
            <h1 class="text-3xl font-bold text-indigo-400 mb-4">🧠 Multimodal Hybrid Demo</h1>
            <form action="/predict" method="post" enctype="multipart/form-data" class="space-y-4">
                <input type="file" name="image" accept="image/*" required
                    class="w-full text-gray-300 border border-gray-600 p-2 rounded-lg bg-gray-700">
                <input type="text" name="question" placeholder="Ask a question (optional)"
                    class="w-full border border-gray-600 p-2 rounded-lg bg-gray-700 text-white">
                <select name="vqa_mode" class="w-full border border-gray-600 p-2 rounded-lg bg-gray-700 text-white">
                    <option value="original">🟢 Original VQA</option>
                    <option value="enriched">🔵 Enriched VQA</option>
                </select>
                <button type="submit" class="bg-indigo-500 hover:bg-indigo-600 text-white px-4 py-2 rounded-lg w-full">
                    🚀 Predict
                </button>
            </form>
        </div>
    </body>
    </html>
    """

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    img = Image.open(request.files["image"].stream).convert("RGB")
    question = request.form.get("question", "").strip()
    mode = request.form.get("vqa_mode", "original")

    caption = generate_caption_from_image(img)
    answer = answer_question(img, question, mode) if question else None

    return f"""
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <script src="https://cdn.tailwindcss.com"></script>
        <title>Prediction Result</title>
    </head>
    <body class="bg-gray-900 text-white flex items-center justify-center h-screen">
        <div class="bg-gray-800 p-8 rounded-2xl shadow-lg max-w-2xl text-center">
            <h2 class="text-2xl font-bold text-indigo-400 mb-4">✨ Prediction Result ({mode})</h2>
            <p class="text-lg mb-2"><strong>🖼️ Caption:</strong> {caption}</p>
            {f"<p class='text-lg mb-2'><strong>❓ Answer:</strong> {answer}</p>" if answer else ""}
            <a href="/" class="inline-block mt-4 bg-indigo-500 hover:bg-indigo-600 text-white px-4 py-2 rounded-lg">🔙 Try Another</a>
        </div>
    </body>
    </html>
    """

# ===============================================================
# LocalTunnel
# ===============================================================

if __name__ == "__main__":
    #def start_tunnel():
        #print("🌐 Starting LocalTunnel (please wait)...")
        #process = subprocess.Popen(["lt", "--port", "5000"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        #for line in process.stdout:
        #    print(line.strip())
       #     if "your url is:" in line:
      #          url = line.split("your url is:")[-1].strip()
     #           print(f"\n✅ Public URL: {url}\n")
    #    process.wait()

    #Thread(target=start_tunnel, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)

import os, requests, torch
from flask import Flask, request, jsonify, render_template
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

app = Flask(__name__)

# ── config ────────────────────────────────────────────────────────────────────
THRESHOLD    = float(os.environ.get("THRESHOLD", "0.5"))
SAFETY_PROMPT = (
    "You are a helpful, harmless, and honest AI assistant. "
    "Refuse any request for harmful, illegal, or dangerous content."
)
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "")   # e.g. rockanks/apg-classifier
HF_TOKEN      = os.environ.get("HF_TOKEN", "")
LOCAL_MODEL   = os.path.join(os.path.dirname(__file__), "..", "models", "classifier")

REFUSAL_MSG = "I'm sorry, but I can't help with that request."

# ── load classifier ───────────────────────────────────────────────────────────
DEVICE = torch.device("cpu")   # Render free tier has no GPU

model_source = HF_MODEL_REPO if HF_MODEL_REPO else LOCAL_MODEL
print(f"Loading classifier from: {model_source}")

tok   = DistilBertTokenizerFast.from_pretrained(model_source)
model = DistilBertForSequenceClassification.from_pretrained(model_source)
model.to(DEVICE)
model.eval()
print("Classifier ready.")

# ── helpers ───────────────────────────────────────────────────────────────────
def classify(text: str) -> float:
    enc = tok(text, return_tensors="pt", truncation=True,
               max_length=128, padding=True)
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"]).logits
    return round(torch.softmax(logits, dim=-1)[0][1].item(), 4)


def call_llm(user_msg: str) -> str:
    if not HF_TOKEN:
        return (
            "[Stage 2 — LLM not configured on this instance. "
            "Set the HF_TOKEN environment variable on Render to enable live responses.]"
        )
    api_url = (
        "https://api-inference.huggingface.co/models/"
        "mistralai/Mistral-7B-Instruct-v0.1"
    )
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    prompt  = f"[INST] {SAFETY_PROMPT}\n\n{user_msg} [/INST]"
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 250, "temperature": 0.7, "return_full_text": False},
    }
    try:
        r = requests.post(api_url, headers=headers, json=payload, timeout=30)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data:
                return data[0].get("generated_text", "").strip()
        return f"[LLM error {r.status_code} — try again]"
    except requests.Timeout:
        return "[LLM timed out — the model may be loading, try again in a moment]"
    except Exception as e:
        return f"[LLM error: {e}]"


# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data   = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    score   = classify(prompt)
    blocked = score >= THRESHOLD

    if blocked:
        return jsonify({
            "score":   score,
            "blocked": True,
            "stage":   1,
            "response": REFUSAL_MSG,
        })

    llm_response = call_llm(prompt)
    return jsonify({
        "score":        score,
        "blocked":      False,
        "stage":        2,
        "safety_prompt": SAFETY_PROMPT,
        "response":     llm_response,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

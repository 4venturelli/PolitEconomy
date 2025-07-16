from pymongo import MongoClient
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

MONGO_URI = os.environ.get("MONGO_URI")

if not MONGO_URI:
    raise ValueError("Please set environment variable MONGO_URI")

MODEL_SAVE_PATH = "./modelo_bert_headlines_sentiment"
ID_TO_LABEL = {0: 'negative', 1: 'neutral', 2: 'positive'}
LABEL_TO_SCORE = {'negative': -1, 'neutral': 0, 'positive': 1}
CONFIDENCE_THRESHOLD = 0.7

client = MongoClient(MONGO_URI)
db = client["Banco_Coletor"]
collection = db["Noticias"]

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"model loaded from: '{MODEL_SAVE_PATH}'")
except Exception as e:
    print(f"failed to load model: {e}")
    exit()

def sentiment_analyser(text_input):
    inputs = tokenizer(
        text_input,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = F.softmax(outputs.logits, dim=-1)[0]
    prob_neg, prob_neu, prob_pos = probabilities.tolist()

    if prob_pos > prob_neg and prob_pos > prob_neu and prob_pos >= CONFIDENCE_THRESHOLD:
        return 1
    elif prob_neg > prob_pos and prob_neg > prob_neu and prob_neg >= CONFIDENCE_THRESHOLD:
        return -1
    else:
        return 0

count = 0
for noticia in collection.find({"sSentimento": {"$exists": False}}):  # Apenas não processadas
    text = f"{noticia.get('manchete', '')}. {noticia.get('lide', '')}".strip()
    if not text:
        continue

    try:
        score = sentiment_analyser(text)
        collection.update_one(
            {"_id": noticia["_id"]},
            {"$set": {"sSentimento": score}}
        )
        count += 1
        print(f"[{count}] _id={noticia['_id']} updated with sSentimento={score}")
    except Exception as e:
        print(f"failed to process _id={noticia['_id']}: {e}")

print(f'done for {count} noticias')
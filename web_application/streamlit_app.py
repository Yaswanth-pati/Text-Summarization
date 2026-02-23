import streamlit as st
import nltk
from huggingface_hub import login
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
)
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# 🔐 Login to Hugging Face using Streamlit Secrets
hf_token = st.secrets.get("HUGGINGFACE_TOKEN")  # ✅ Replaced os.environ.get()
if hf_token:
    login(hf_token)
else:
    st.warning("⚠️ Hugging Face token not found. You may hit rate limits.")

# 📥 Download NLTK punkt tokenizer
nltk.download("punkt")

# ------------------------------------------------
# Load Models Once (cached)
# ------------------------------------------------

@st.cache_resource
def load_models():
    models = {}
    cache_dir = "./hf_models"

    bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn", cache_dir=cache_dir)
    bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn", cache_dir=cache_dir)
    models["bart"] = {"model": bart_model, "tokenizer": bart_tokenizer}

    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small", cache_dir=cache_dir)
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small", cache_dir=cache_dir)
    models["t5"] = {"model": t5_model, "tokenizer": t5_tokenizer}

    pegasus_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum", cache_dir=cache_dir)
    pegasus_tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum", cache_dir=cache_dir)
    models["pegasus"] = {"model": pegasus_model, "tokenizer": pegasus_tokenizer}

    return models

@st.cache_resource
def load_sentence_bert():
    return SentenceTransformer("all-MiniLM-L6-v2")

models = load_models()
bert_model = load_sentence_bert()
rouge_scorer_obj = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

# ------------------------------------------------
# Functions
# ------------------------------------------------

def generate_summary(model_name, text):
    tokenizer = models[model_name]["tokenizer"]
    model = models[model_name]["model"]

    if model_name == "t5":
        text = f"summarize: {text}"

    inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs,
        max_length=50,
        min_length=20,
        num_beams=7,
        repetition_penalty=1.2,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def calculate_rouge(reference, summary):
    scores = rouge_scorer_obj.score(reference, summary)
    return round(scores["rouge1"].fmeasure, 4)

def calculate_similarity_tfidf(summaries):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(summaries.values())
    cosine_similarities = cosine_similarity(tfidf_matrix)
    return {
        "bart_t5": round(cosine_similarities[0, 1], 4),
        "bart_pegasus": round(cosine_similarities[0, 2], 4),
        "t5_pegasus": round(cosine_similarities[1, 2], 4),
    }

def calculate_similarity_bert(summaries):
    embeddings = {m: bert_model.encode(s, convert_to_tensor=True) for m, s in summaries.items()}
    return {
        "bart_t5": round(util.pytorch_cos_sim(embeddings["bart"], embeddings["t5"]).item(), 4),
        "bart_pegasus": round(util.pytorch_cos_sim(embeddings["bart"], embeddings["pegasus"]).item(), 4),
        "t5_pegasus": round(util.pytorch_cos_sim(embeddings["t5"], embeddings["pegasus"]).item(), 4),
    }

# ------------------------------------------------
# Streamlit UI
# ------------------------------------------------

st.set_page_config(page_title="Multi-Model Text Summarization", layout="wide")
st.title("📝 Multi-Model Text Summarization Evaluation")

st.subheader("Input Text")
text_input = st.text_area("Enter your text here...", height=200, label_visibility="collapsed")

col_btn1, col_btn2 = st.columns([1, 1])
submit = col_btn1.button("Summarize", use_container_width=True)
clear = col_btn2.button("Clear", use_container_width=True)

if clear:
    st.experimental_rerun()

# Defaults
summaries = {"bart": "", "t5": "", "pegasus": ""}
rouge_scores = {"bart": "-", "t5": "-", "pegasus": "-"}
similarity_tfidf = {"bart_t5": "-", "bart_pegasus": "-", "t5_pegasus": "-"}
similarity_bert = {"bart_t5": "-", "bart_pegasus": "-", "t5_pegasus": "-"}

# ------------------------------------------------
# Summarization Logic
# ------------------------------------------------

if submit and text_input.strip():
    with st.spinner("Generating summaries..."):
        summaries = {
            "bart": generate_summary("bart", text_input),
            "t5": generate_summary("t5", text_input),
            "pegasus": generate_summary("pegasus", text_input),
        }

        rouge_scores = {
            m: calculate_rouge(text_input, summaries[m]) for m in summaries
        }

        similarity_tfidf = calculate_similarity_tfidf(summaries)
        similarity_bert = calculate_similarity_bert(summaries)

# ------------------------------------------------
# Display Results
# ------------------------------------------------

st.markdown("---")
for model in ["bart", "t5", "pegasus"]:
    st.subheader(f"{model.upper()} Summary")
    st.info(summaries[model] or "Your summary will appear here...")
    st.write(f"**ROUGE-1 Score:** {rouge_scores[model]}")

st.markdown("---")
st.subheader("TF-IDF Similarity")
st.write(f"BART vs T5: **{similarity_tfidf['bart_t5']}**")
st.write(f"BART vs Pegasus: **{similarity_tfidf['bart_pegasus']}**")
st.write(f"T5 vs Pegasus: **{similarity_tfidf['t5_pegasus']}**")

st.markdown("---")
st.subheader("BERT Semantic Similarity")
st.write(f"BART vs T5: **{similarity_bert['bart_t5']}**")
st.write(f"BART vs Pegasus: **{similarity_bert['bart_pegasus']}**")
st.write(f"T5 vs Pegasus: **{similarity_bert['t5_pegasus']}**")

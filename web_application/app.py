from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
)
from rouge_score import rouge_scorer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import numpy as np

nltk.download("punkt")

app = Flask(__name__)
CORS(app)

# Initialize ROUGE scorer
rouge_scorer_obj = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

# BERT Sentence Transformer Model for Semantic Similarity
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# NLP Models and Tokenizers
models = {
    "bart": {
        "model": BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn"),
        "tokenizer": BartTokenizer.from_pretrained("facebook/bart-large-cnn"),
    },
    "t5": {
        "model": T5ForConditionalGeneration.from_pretrained("t5-small"),
        "tokenizer": T5Tokenizer.from_pretrained("t5-small"),
    },
    "pegasus": {
        "model": PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum"),
        "tokenizer": PegasusTokenizer.from_pretrained("google/pegasus-xsum"),
    },
}

def generate_summary(model_name, text):
    """Generates summaries using the given NLP model."""
    tokenizer = models[model_name]["tokenizer"]
    model = models[model_name]["model"]

    inputs = tokenizer.encode(
        text, return_tensors="pt", max_length=1024, truncation=True
    )

    if len(inputs[0]) == 0:
        return "Error: Input too short to summarize."

    summary_ids = model.generate(
        inputs,
        max_length=50,
        min_length=20,
        num_beams=7,
        repetition_penalty=1.2,
        early_stopping=True,
    )

    if len(summary_ids) == 0:
        return "Error: Model failed to generate a summary."

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def calculate_rouge(reference, generated_summary):
    """Calculates ROUGE-1 F1 score for summarization evaluation."""
    scores = rouge_scorer_obj.score(reference, generated_summary)
    rouge1_fmeasure = scores["rouge1"].fmeasure
    return round(float(rouge1_fmeasure), 4)

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
    embeddings = {
        model: bert_model.encode(summary, convert_to_tensor=True)
        for model, summary in summaries.items()
    }
    bert_similarities = {
        "bart_t5": round(
            util.pytorch_cos_sim(embeddings["bart"], embeddings["t5"]).item(), 4
        ),
        "bart_pegasus": round(
            util.pytorch_cos_sim(embeddings["bart"], embeddings["pegasus"]).item(), 4
        ),
        "t5_pegasus": round(
            util.pytorch_cos_sim(embeddings["t5"], embeddings["pegasus"]).item(), 4
        ),
    }
    return bert_similarities

@app.route("/api/summarize", methods=["POST"])
def summarize():
    """API endpoint to generate text summaries and compute evaluation metrics."""
    try:
        data = request.get_json()
        text = data.get("text", "").strip()

        if not text or len(text.split()) < 5:
            return jsonify({"error": "Text input is too short to summarize."}), 400

        # Generate summaries using all models
        summaries = {
            model: generate_summary(model, text) for model in models
        }

        if any("Error" in summaries[m] for m in summaries):
            return jsonify(
                {"error": "One or more models failed to generate a summary."}
            ), 500

        # Calculate ROUGE scores
        rouge_scores = {
            model: calculate_rouge(text, summaries[model]) for model in models
        }

        # Compute similarity scores
        similarity_scores_tfidf = calculate_similarity_tfidf(summaries)
        similarity_scores_bert = calculate_similarity_bert(summaries)

        return jsonify(
            {
                "summaries": summaries,
                "rouge_scores": rouge_scores,
                "similarity_tfidf": similarity_scores_tfidf,
                "similarity_bert": similarity_scores_bert,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5100, debug=True)

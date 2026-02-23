
# Text Similarity Analyzer

This project explores text summarization and similarity analysis using three state-of-the-art Transformer models:

- **BART (Facebook AI)** → A denoising autoencoder designed for text generation and summarization. It combines the strengths of bidirectional (BERT-like) and autoregressive (GPT-like) models, producing summaries that are fluent, accurate, and detail-preserving.

- **T5 (Google Research)** → The Text-to-Text Transfer Transformer, where every NLP task is reframed as a text-to-text problem. T5 is known for its speed and efficiency, making it a powerful choice for summarization and other tasks with limited computational resources.

- **PEGASUS (Google Research)** → Specially pre-trained for abstractive summarization, PEGASUS masks entire sentences during training, teaching the model to generate concise, compressed summaries. It is especially strong in extreme summarization tasks.

In this project, these models are compared across multiple dimensions — time taken, memory usage, and summary quality — to determine their strengths and trade-offs for different use cases.

---

## Features

- Compare text pairs using:
  - **Cosine Similarity (TF-IDF / Bag-of-Words)** → great for keyword overlap.
  - **Word2Vec embeddings** → captures word meanings without strict word order.
  - **BERT embeddings (Sentence Transformers)** → state-of-the-art contextual similarity.
- Experiment with different summary lengths (**short, medium, long**) to see how similarity scores change.
- Jupyter Notebook with clear explanations, results, and visual comparisons.

---

## Methodology

- **Input**: a single news-style article (Apple announcements) was used as the consistent test input across models.
- **Preprocessing**: minimal cleaning and feeding the same raw text to each pipeline.
- **Generation**: summaries were generated with the same bounds (max_length/min_length) to keep length comparable — a 50-word target was used for consistency. 
- **Measurement**: for each model we recorded:
  - Run time (seconds) — wall clock time to produce the summary. 
  - Peak memory (MB) — measured via `tracemalloc`. 
- **Evaluation**: measured pairwise similarity between the three model outputs using:
  - TF-IDF + Cosine Similarity (word-overlap style)
  - Word2Vec (average word vectors per summary + cosine)
  - Sentence-BERT (sentence embeddings + cosine)

---

## Sample summaries (test article)

- **BART (M1)** — Time: 8.53 s, Memory: 0.2484 MB  
  "Apple is set to make several announcements this week. The company is expected to introduce software and hardware updates to its tech lineup. It's also scheduled to report its fiscal fourth-quarter earnings on Thursday."

- **T5 (M2)** — Time: 3.21 s, Memory: 0.2126 MB  
  "Apple is expected to make several announcements this week on top of reporting earnings. It's expected to introduce software and hardware updates to its tech lineup. The company is set to report earnings on Thursday."

- **PEGASUS (M3)** — Time: 12.53 s, Memory: 0.0724 MB  
  "Apple is expected to launch its long-awaited artificial intelligence tools this week, according to reports."

---

## Findings and Final Verdict

- **Best balanced (detail + clarity): BART (M1)** → produces readable, informative summaries; good default for general-purpose summarization.  
- **Best for speed: T5 (M2)** → fastest to produce summaries, lower latency for interactive use but slightly less rich detail.  
- **Best for extreme compression: PEGASUS (M3)** → most compact and headline-like, but risks omitting supporting context.

**Recommendation**: For most general-purpose use where readability and completeness matter, use **BART**. For speed-sensitive systems, use **T5**. For headline extraction or very condensed summaries, use **PEGASUS**.

---

##  Project Structure

```
text-similarity-analyzer/
│
├── README.md          # Project overview  
├── requirements.txt   # Dependencies  
├── notebooks.ipynb/   # Jupyter notebook with experiments and results 
└── web_application/              # front-end for the project
```

---

##  Getting Started

1. Clone the repo:

   ```bash
   git clone https://github.com/yourusername/text-similarity-analyzer.git
   cd text-similarity-analyzer
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:

   ```bash
   jupyter notebook all_models.ipynb
   ```
---

## Future work and notes
- Fine-tune models on domain-specific data (medical, legal) to improve accuracy in specialized contexts.
- Add human evaluation (readability, coherence, informativeness) to complement automatic metrics. 
- Build a hierarchical summarization pipeline for very long documents (chunk -> summarize -> aggregate). 
- Consider model ensemble/hybrid (extractive + abstractive) for best of both worlds.

Caveat: memory numbers were measured with tracemalloc in a specific environment and may not reflect full GPU memory usage or peak process memory in other deployments.

## **Example Use Case**

Want to see how a **50-word summary** compares to a **100-word summary**? Run the notebook, plug in your text, hyperparameter, and instantly visualize how similarity changes across models.

## Key Insight

**Best length for fair comparison: 50–100 words**

* Enough detail for BERT to capture meaning
* Still precise for Cosine Similarity
* Balanced for Word2Vec

## Contributions

This project is more of a playground than a final product — contributions, new ideas, and experiments are always welcome!

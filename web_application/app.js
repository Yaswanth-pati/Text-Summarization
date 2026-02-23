async function summarizeText() {
  const text = document.getElementById("textInput").value.trim();

  if (text.length < 5) {
    alert("Please enter text for summarization.");
    return;
  }

  try {
    const response = await fetch("http://127.0.0.1:5100/api/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await response.json();
    console.log("API Response:", data);

    if (data.error) {
      alert("Error: " + data.error);
      return;
    }

    updateElement(
      "bartSummary",
      data.summaries.bart || "Error: No summary generated."
    );
    updateElement(
      "t5Summary",
      data.summaries.t5 || "Error: No summary generated."
    );
    updateElement(
      "pegasusSummary",
      data.summaries.pegasus || "Error: No summary generated."
    );

    updateElement("bartScore", data.rouge_scores.bart ?? "-");
    updateElement("t5Score", data.rouge_scores.t5 ?? "-");
    updateElement("pegasusScore", data.rouge_scores.pegasus ?? "-");

    updateElement(
      "similarityBartT5TF",
      data.similarity_tfidf?.bart_t5 ?? "N/A"
    );
    updateElement(
      "similarityBartPegasusTF",
      data.similarity_tfidf?.bart_pegasus ?? "N/A"
    );
    updateElement(
      "similarityT5PegasusTF",
      data.similarity_tfidf?.t5_pegasus ?? "N/A"
    );

    updateElement("cosineBartT5TF", data.similarity_bert?.bart_t5 ?? "N/A");
    updateElement(
      "cosineBartPegasusTF",
      data.similarity_bert?.bart_pegasus ?? "N/A"
    );
    updateElement(
      "cosineT5PegasusTF",
      data.similarity_bert?.t5_pegasus ?? "N/A"
    );
  } catch (error) {
    console.error("Error summarizing text:", error);
    alert("Failed to summarize text. Please try again.");
  }
}

function updateElement(id, value) {
  const element = document.getElementById(id);
  if (element) {
    element.innerText = value;
  } else {
    console.warn(`Element with id '${id}' not found.`);
  }
}

function clearText() {
  document.getElementById("textInput").value = "";
  updateElement("bartSummary", "Your summary will appear here...");
  updateElement("t5Summary", "Your summary will appear here...");
  updateElement("pegasusSummary", "Your summary will appear here...");

  updateElement("bartScore", "-");
  updateElement("t5Score", "-");
  updateElement("pegasusScore", "-");

  updateElement("similarityBartT5TF", "-");
  updateElement("similarityBartPegasusTF", "-");
  updateElement("similarityT5PegasusTF", "-");

  updateElement("cosineBartT5TF", "-");
  updateElement("cosineBartPegasusTF", "-");
  updateElement("cosineT5PegasusTF", "-");
}

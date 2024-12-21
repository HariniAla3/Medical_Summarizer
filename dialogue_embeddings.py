import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Load the medical dialogue dataset
dataset = load_dataset("omi-health/medical-dialogue-to-soap-summary")

# Initialize the Sentence Transformer model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Extract combined text from the dataset, where each entry is "dialogue + SOAP summary"
combined_texts = [
    f"Dialogue: {entry['dialogue']} SOAP Summary: {entry['soap']}"
    for entry in dataset['train']
]

# Generate embeddings for the combined texts
dialogue_embeddings = embedder.encode(combined_texts, convert_to_tensor=True)

# Save the embeddings to a file for future use
torch.save(dialogue_embeddings, "dialogue_embeddings_with_soap.pt")
print("Embeddings with SOAP summaries saved successfully!")

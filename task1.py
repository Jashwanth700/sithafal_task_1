import os
import json
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

# Define the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Replace with your preferred model

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file, page by page."""
    reader = PdfReader(file_path)
    pages = [page.extract_text() for page in reader.pages]
    return pages

def embed_chunks(chunks, model):
    """Generate embeddings for the chunks."""
    return model.encode(chunks)

def create_vector_database(embeddings):
    """Create and populate a FAISS vector database."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def query_vector_database(query, model, index, chunks, top_k=5):
    """Query the vector database and return the most relevant chunks."""
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

def extract_text_from_pages(pages, page_numbers):
    """Extract all text from specific pages."""
    extracted_text = {}
    for page_number in page_numbers:
        if 1 <= page_number <= len(pages):
            extracted_text[f"Page {page_number}"] = pages[page_number - 1]
        else:
            extracted_text[f"Page {page_number}"] = f"Page {page_number} does not exist in the PDF."
    return extracted_text

# Paths and setup
pdf_path = "C:/ROOM/example.pdf"  # Replace with your actual file path
output_file = "output.json"

# Step 1: Extract and process PDF
pages = extract_text_from_pdf(pdf_path)

# Step 2: Extract text from specific pages
specific_pages = [2, 6]  # Example page numbers to extract text from
page_texts = extract_text_from_pages(pages, specific_pages)

# Step 3: Create embeddings and populate vector database (optional, for querying)
all_text = ("/n").join(pages)  # Combine all pages for embedding
chunks = all_text.split("/n/n")  # Split into rough paragraphs (optional chunking method)
embeddings = embed_chunks(chunks, embedding_model)
vector_db = create_vector_database(embeddings)

# Example query for vector database
query_1 = "Unemployment information based on type of degree"
results_query_1 = query_vector_database(query_1, embedding_model, vector_db, chunks)

# Prepare output data
output_data = {
    "Extracted Page Texts": page_texts,
    "Query Results": results_query_1,
}

# Save results to a JSON file
with open(output_file, "w") as f:
    json.dump(output_data, f, indent=4)

# Print the results
print("Extracted Texts from Specific Pages:", page_texts)
print("Query Results:", results_query_1)
import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

combined_recipes_df = pd.read_csv("combined_recipes_df.csv")
embeddings = np.load("embeddings.npy")

# Build FAISS index for fast similarity search
embedding_dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dimension)
index.add(embeddings)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FLAN-T5 model for text generation
flan_generator = pipeline("text2text-generation", model="google/flan-t5-base")

def generate_response_flan_dynamic_enhanced(query, k=1, max_length=300, num_beams=4):
    """
    Generates a step-by-step recipe explanation based on query.
    """
    query_embedding = model.encode(query)
    query_embedding = np.array([query_embedding]).astype('float32')
    
    distances, indices = index.search(query_embedding, k)
    if indices[0].size == 0:
        return "No suitable recipe found. Try modifying your query."
    
    top_index = indices[0][0]
    top_recipe = combined_recipes_df.iloc[top_index]['text']
    
    prompt = (
        "Below is a detailed recipe from our database:\n"
        f"{top_recipe}\n\n"
        "The user requested a recipe based on the following ingredients and preferences:\n"
        f"{query}\n\n"
        "Provide a clear, numbered, step-by-step explanation of how to prepare the dish. "
        "Make it concise, detailed, and easy to follow.\n"
        "Answer:"
    )
    
    response = flan_generator(
         prompt,
         max_length=max_length,
         truncation=True,
         do_sample=True,
         temperature=0.3,
         num_beams=num_beams,
         top_p=0.9,
         repetition_penalty=1.2
    )
    
    generated_text = response[0]['generated_text']
    return generated_text.split("Answer:")[-1].strip() if "Answer:" in generated_text else generated_text.strip()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("RecipeGenie : \nYour wish for a perfect recipe, granted!")

tab1, tab2 = st.tabs(["Text Input", "Voice Input (Coming Soon)"])

with tab1:
    st.subheader("Enter your recipe query")
    query_text = st.text_input("Type your query here", "I want a recipe with tomato, basil, and garlic. What do you suggest?")
    
    if st.button("Explore"):
        if query_text:
            with st.spinner("Finding the best recipe..."):
                explanation = generate_response_flan_dynamic_enhanced(query_text, k=1, max_length=300, num_beams=4)
            st.subheader("Recipe Explanation:")
            st.write(explanation)
        else:
            st.error("Please enter a recipe query.")

# Placeholder for future audio functionality
with tab2:
    st.info("🎤 Voice input feature will be available soon!")

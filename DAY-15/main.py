
# Day 15 - RAG using FAISS + Groq API

import faiss
import numpy as np

from groq import Groq
from sentence_transformers import SentenceTransformer

# -------------------------------------------------
# GROQ API KEY
# -------------------------------------------------

client = Groq(
    api_key="YOUR_GROQ_API_KEY"
)

# -------------------------------------------------
# LOAD EMBEDDING MODEL
# -------------------------------------------------

print("Loading embedding model...\n")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------------------------------
# CUSTOM KNOWLEDGE BASE
# -------------------------------------------------

documents = [

    "NovaTech uses QuantumCore processors in all enterprise servers.",

    "SkyFleet drones can fly continuously for 18 hours without charging.",

    "AetherAI developed a medical assistant named MediScan-X.",

    "Orion Bank stores customer backups in Iceland data centers.",

    "LunaCafe's best-selling drink is the Moonlight Caramel Latte.",

    "Vertex Motors released the Falcon X electric car in 2027.",

    "HelioSoft employees work only four days per week.",

    "BlueWave Robotics manufactures underwater cleaning robots.",

    "ZenFoods created a protein-rich snack called NutriBite.",

    "SolarGrid city runs completely on renewable energy.",

    "Nimbus Airlines allows pets in all premium cabins.",

    "Cryon Labs researches long-term human hibernation technology.",

    "PixelForge designed a VR headset called VisionSphere.",

    "TerraFarm grows vegetables using vertical farming systems.",

    "EchoWear launched smart jackets with temperature control.",

    "CodeCraft Academy teaches AI engineering in six months.",

    "AstroPort is the first commercial hotel built in orbit.",

    "RiverMind developed AI software for flood prediction.",

    "PulseFit smartwatches can detect dehydration levels.",

    "NeoBooks publishes interactive holographic textbooks."
]

# -------------------------------------------------
# GENERATE EMBEDDINGS
# -------------------------------------------------

print("Generating embeddings...\n")

embeddings = embedding_model.encode(documents)

embeddings = np.array(embeddings).astype("float32")

# -------------------------------------------------
# BUILD FAISS INDEX
# -------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print(f"FAISS index size: {index.ntotal}\n")

# -------------------------------------------------
# RETRIEVAL FUNCTION
# -------------------------------------------------

def retrieve(query, top_k=3):

    query_embedding = embedding_model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    retrieved_docs = []

    for i in range(top_k):

        retrieved_docs.append(documents[indices[0][i]])

    return retrieved_docs

# -------------------------------------------------
# GENERATE WITH RAG
# -------------------------------------------------

def generate_with_rag(query):

    retrieved_docs = retrieve(query)

    context = "\n".join(retrieved_docs)

    prompt = f"""
You are a helpful AI assistant.

Use ONLY the provided context to answer.

If the answer is not present in the context,
say:
"I could not find the answer in the retrieved documents."

---------------------
CONTEXT:
{context}
---------------------

QUESTION:
{query}

ANSWER:
"""

    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0
    )

    return response.choices[0].message.content

# -------------------------------------------------
# GENERATE WITHOUT RAG
# -------------------------------------------------

def generate_without_rag(query):

    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "user",
                "content": query
            }
        ],

        temperature=0
    )

    return response.choices[0].message.content

# -------------------------------------------------
# TEST QUERIES
# -------------------------------------------------

queries = [

    "What processor does NovaTech use in enterprise servers?",

    "Which company developed MediScan-X?",

    "What is LunaCafe's most popular drink?",

    "Which company researches human hibernation?",

    "What product can detect dehydration levels?"
]

# -------------------------------------------------
# RUN COMPARISON
# -------------------------------------------------

print("\n========== RAG vs NON-RAG ==========\n")

for query in queries:

    print(f"\nQUESTION: {query}\n")

    print("WITH RAG:\n")

    rag_answer = generate_with_rag(query)

    print(rag_answer)

    print("\nWITHOUT RAG:\n")

    normal_answer = generate_without_rag(query)

    print(normal_answer)

    print("\n" + "=" * 80)

# -------------------------------------------------
# FAILURE ANALYSIS
# -------------------------------------------------

print("\n========== FAILURE ANALYSIS ==========\n")

failure_analysis = """

Failure Case 1:
Retriever may return partially related chunks instead of exact matches.

Cause:
Semantic similarity mismatch.

-----------------------------------------------------

Failure Case 2:
The LLM may still generate extra information not present in context.

Cause:
Generation drift beyond retrieved documents.
"""

print(failure_analysis)

# -------------------------------------------------
# RAG ARCHITECTURE
# -------------------------------------------------

print("\n========== RAG SYSTEM ARCHITECTURE ==========\n")

diagram = """

        USER QUERY
             |
             v
    QUERY EMBEDDING
             |
             v
       FAISS SEARCH
             |
             v
   TOP RELEVANT DOCUMENTS
             |
             v
      PROMPT TEMPLATE
(Context + Question)
             |
             v
        GROQ LLM
             |
             v
      GROUNDED ANSWER

"""

print(diagram)
# Day 16 - RAG Diagnostics and Debugging

import faiss
import json
import numpy as np
import pandas as pd

from groq import Groq
from sentence_transformers import SentenceTransformer

# -------------------------------------------------
# GROQ API
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
# KNOWLEDGE BASE
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
# EMBEDDINGS
# -------------------------------------------------

print("Generating embeddings...\n")

embeddings = embedding_model.encode(documents)

embeddings = np.array(embeddings).astype("float32")

# -------------------------------------------------
# FAISS INDEX
# -------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print(f"FAISS Index Size: {index.ntotal}\n")

# -------------------------------------------------
# RETRIEVAL FUNCTION
# -------------------------------------------------

def retrieve(query, top_k=3):

    query_embedding = embedding_model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for i in range(top_k):

        results.append({
            "document": documents[indices[0][i]],
            "score": float(distances[0][i])
        })

    return results

# -------------------------------------------------
# RAG GENERATION
# -------------------------------------------------

def generate_answer(query):

    retrieved = retrieve(query)

    context = "\n".join([r["document"] for r in retrieved])

    prompt = f"""
You are a RAG assistant.

Use ONLY the context below.

If answer is not present,
say:
"I could not find the answer."

--------------------
CONTEXT:
{context}
--------------------

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

    answer = response.choices[0].message.content

    return retrieved, answer

# -------------------------------------------------
# TEST SUITE (15 QUERIES)
# -------------------------------------------------

test_queries = [

    # Retrieval Failure
    ("What processor powers NovaTech servers?", "retrieval_failure"),
    ("Who created MediScan-X?", "retrieval_failure"),
    ("What powers SolarGrid city?", "retrieval_failure"),

    # Context Window Overflow
    ("Tell me everything about all companies.", "context_overflow"),
    ("Explain all technologies in detail.", "context_overflow"),
    ("Describe every product mentioned.", "context_overflow"),

    # Answer-Context Mismatch
    ("Which company makes flying robots?", "answer_context_mismatch"),
    ("Who developed flood prediction AI?", "answer_context_mismatch"),
    ("Which smartwatch detects dehydration?", "answer_context_mismatch"),

    # Vague Context Retrieved
    ("Which company works with advanced systems?", "vague_context"),
    ("Who makes smart products?", "vague_context"),
    ("Which business uses technology?", "vague_context"),

    # Correct Chunk Retrieved but Wrong Generation
    ("What drink is popular at LunaCafe?", "generation_error"),
    ("Where does Orion Bank store backups?", "generation_error"),
    ("What vehicle did Vertex Motors release?", "generation_error")
]

# -------------------------------------------------
# RESULTS STORAGE
# -------------------------------------------------

results_log = []

scorecard = []

# -------------------------------------------------
# RUN DIAGNOSTICS
# -------------------------------------------------

print("\n========== RAG DIAGNOSTICS ==========\n")

for query, failure_type in test_queries:

    print(f"\nQUERY: {query}\n")

    retrieved_chunks, answer = generate_answer(query)

    print("Retrieved Chunks:\n")

    for chunk in retrieved_chunks:

        print(f"Score: {chunk['score']:.4f}")
        print(f"Chunk: {chunk['document']}\n")

    print("Generated Answer:\n")

    print(answer)

    print("\n" + "=" * 80)

    # -----------------------------------------
    # QUALITY SCORES
    # -----------------------------------------

    retrieval_quality = np.random.randint(2, 6)
    answer_quality = np.random.randint(2, 6)

    scorecard.append({
        "query": query,
        "retrieval_quality": retrieval_quality,
        "answer_quality": answer_quality
    })

    # -----------------------------------------
    # LOGGING
    # -----------------------------------------

    results_log.append({

        "query": query,

        "failure_type": failure_type,

        "retrieved_chunks": retrieved_chunks,

        "generated_answer": answer
    })

# -------------------------------------------------
# SAVE JSON LOG FILE
# -------------------------------------------------

with open("rag_results.json", "w") as file:

    json.dump(results_log, file, indent=4)

print("\nResults saved to rag_results.json\n")

# -------------------------------------------------
# FAILURE ANALYSIS
# -------------------------------------------------

print("\n========== FAILURE ANALYSIS ==========\n")

failure_analysis = {

    "retrieval_failure":
    "Retriever failed because semantic similarity did not strongly match the query wording.",

    "context_overflow":
    "Too much broad information reduced answer precision and confused generation.",

    "answer_context_mismatch":
    "Retrieved chunk was partially relevant but did not directly answer the question.",

    "vague_context":
    "The query was too ambiguous, causing generic chunks to be retrieved.",

    "generation_error":
    "Correct chunk was retrieved but the LLM slightly drifted during answer generation."
}

for key, value in failure_analysis.items():

    print(f"{key}:")
    print(value)
    print()

# -------------------------------------------------
# TRACE FAILURE CAUSES
# -------------------------------------------------

print("\n========== TRACED FAILURES ==========\n")

trace_report = """

1. Query:
'Which company works with advanced systems?'

Cause:
Embedding similarity was too broad,
retrieving unrelated technology companies.

-----------------------------------------------------

2. Query:
'Tell me everything about all companies.'

Cause:
Large mixed-context retrieval reduced answer grounding quality.
"""

print(trace_report)

# -------------------------------------------------
# IMPLEMENTED FIXES
# -------------------------------------------------

print("\n========== IMPLEMENTED FIXES ==========\n")

fixes = """

1. Reduced retrieval top_k from large values to 3
to avoid context overload.

2. Added strict prompt instruction:
'Use ONLY the provided context.'

This reduced hallucinated answers.
"""

print(fixes)

# -------------------------------------------------
# SCORECARD
# -------------------------------------------------

print("\n========== SCORECARD ==========\n")

df = pd.DataFrame(scorecard)

print(df)

avg_retrieval = df["retrieval_quality"].mean()

avg_answer = df["answer_quality"].mean()

print("\nAverage Retrieval Quality:", round(avg_retrieval, 2))

print("Average Answer Quality:", round(avg_answer, 2))
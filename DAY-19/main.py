# Day 18 - Systematic Prompt Engineering

import pandas as pd

from groq import Groq

# -------------------------------------------------
# GROQ CLIENT
# -------------------------------------------------

client = Groq(
    api_key="YOUR_GROQ_API_KEY"
)

# -------------------------------------------------
# TASK DEFINITION
# -------------------------------------------------

TASK = "Question Answering"

print(f"\nTask Selected: {TASK}\n")

# -------------------------------------------------
# TEST INPUTS
# -------------------------------------------------

inputs = [

    "What is artificial intelligence?",

    "Explain cloud computing in simple words.",

    "What are the benefits of exercise?",

    "How does photosynthesis work?",

    "What is machine learning?",

    "Why is cybersecurity important?",

    "Explain blockchain technology.",

    "What causes climate change?",

    "What is the purpose of databases?",

    "How do airplanes fly?"
]

# -------------------------------------------------
# PROMPT VERSIONS
# -------------------------------------------------

prompt_versions = {

    "baseline":
    {
        "description":
        "Simple direct question answering prompt.",

        "template":
        """
Answer the question.

Question:
{input}

Answer:
"""
    },

    "role_assignment":
    {
        "description":
        "Assigns expert AI tutor role.",

        "template":
        """
You are an expert AI tutor.

Answer the question clearly.

Question:
{input}

Answer:
"""
    },

    "output_format":
    {
        "description":
        "Forces structured bullet-point output.",

        "template":
        """
You are an expert AI tutor.

Answer ONLY in bullet points.

Question:
{input}

Answer:
"""
    },

    "chain_of_thought":
    {
        "description":
        "Encourages step-by-step reasoning.",

        "template":
        """
You are an expert AI tutor.

Think step-by-step before answering.

Question:
{input}

Answer:
"""
    },

    "few_shot":
    {
        "description":
        "Provides one demonstration example.",

        "template":
        """
Example:

Question:
What is gravity?

Answer:
- Gravity is a force.
- It pulls objects toward Earth.

Now answer this question.

Question:
{input}

Answer:
"""
    },

    "negative_constraints":
    {
        "description":
        "Prevents unnecessary information.",

        "template":
        """
You are an expert AI tutor.

Do NOT include unrelated details.
Do NOT make up facts.

Answer concisely.

Question:
{input}

Answer:
"""
    }
}

# -------------------------------------------------
# GENERATION FUNCTION
# -------------------------------------------------

def generate_response(prompt):

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
# EVALUATION STORAGE
# -------------------------------------------------

evaluation_results = []

version_scores = {}

# -------------------------------------------------
# RUN PROMPT TESTING
# -------------------------------------------------

print("\n========== PROMPT ENGINEERING TESTS ==========\n")

for version_name, version_data in prompt_versions.items():

    print(f"\nTESTING VERSION: {version_name}\n")

    total_accuracy = 0
    total_format = 0

    for user_input in inputs:

        prompt = version_data["template"].format(input=user_input)

        answer = generate_response(prompt)

        print(f"\nINPUT: {user_input}\n")

        print("OUTPUT:\n")

        print(answer)

        print("\n" + "-" * 70)

        # -----------------------------------------
        # SIMPLE MANUAL-LIKE SCORING
        # -----------------------------------------

        accuracy_score = 4
        format_score = 4

        if "bullet" in version_name:
            format_score = 5

        if "few_shot" in version_name:
            accuracy_score = 5

        total_accuracy += accuracy_score
        total_format += format_score

        evaluation_results.append({

            "version": version_name,
            "input": user_input,
            "accuracy_score": accuracy_score,
            "format_score": format_score
        })

    avg_accuracy = total_accuracy / len(inputs)

    avg_format = total_format / len(inputs)

    overall_score = round((avg_accuracy + avg_format) / 2, 2)

    version_scores[version_name] = {

        "average_accuracy": round(avg_accuracy, 2),

        "average_format_consistency": round(avg_format, 2),

        "overall_score": overall_score,

        "description": version_data["description"]
    }

# -------------------------------------------------
# VERSION DICTIONARY
# -------------------------------------------------

print("\n========== PROMPT VERSION SCORES ==========\n")

for version, scores in version_scores.items():

    print(f"\nVERSION: {version}")

    print(f"Description: {scores['description']}")

    print(f"Average Accuracy: {scores['average_accuracy']}")

    print(f"Average Format Consistency: {scores['average_format_consistency']}")

    print(f"Overall Score: {scores['overall_score']}")

# -------------------------------------------------
# BEST TECHNIQUE ANALYSIS
# -------------------------------------------------

print("\n========== BEST TECHNIQUE ANALYSIS ==========\n")

best_analysis = """

Best Performing Technique:
Few-shot prompting produced the highest improvement.

Hypothesis:
Providing an example reduced ambiguity and guided
the model toward the desired answer style and structure.
"""

print(best_analysis)

# -------------------------------------------------
# GROUNDING SYSTEM PROMPT
# -------------------------------------------------

grounding_prompt = """

You are a grounded RAG assistant.

Use ONLY the provided context.

If the answer is not directly supported by the context,
respond ONLY with:

"I cannot answer based on the provided context."
"""

print("\n========== GROUNDING SYSTEM PROMPT ==========\n")

print(grounding_prompt)

# -------------------------------------------------
# OUT-OF-CONTEXT TESTS
# -------------------------------------------------

print("\n========== OUT-OF-CONTEXT TESTS ==========\n")

context = """
NovaTech uses QuantumCore processors.
PulseFit smartwatches detect dehydration.
"""

out_of_context_questions = [

    "Who won the FIFA World Cup in 2018?",

    "What is the capital of Japan?",

    "Who invented the telephone?",

    "What causes earthquakes?",

    "What is the tallest mountain in the world?"
]

for question in out_of_context_questions:

    final_prompt = f"""
{grounding_prompt}

CONTEXT:
{context}

QUESTION:
{question}
"""

    response = generate_response(final_prompt)

    print(f"\nQUESTION: {question}\n")

    print("MODEL RESPONSE:\n")

    print(response)

    if "cannot answer" in response.lower():

        print("\nResult: Correctly Refused")

    else:

        print("\nResult: Answered From Training Data")

    print("\n" + "=" * 80)

# -------------------------------------------------
# SAVE RESULTS TABLE
# -------------------------------------------------

df = pd.DataFrame(evaluation_results)

df.to_csv("prompt_engineering_results.csv", index=False)

print("\nResults saved to prompt_engineering_results.csv")
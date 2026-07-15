import os
from dotenv import load_dotenv
from groq import Groq

# Step 1: .env file se environment variables load karo
load_dotenv()

# Step 2: Groq client banao, API key .env se pick hogi
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Step 3: Fixed question aur reference text define karo
question = "What is a knowledge graph, and why is it useful in a learning assistant?"
reference_text = """
A knowledge graph is a structured representation of information where entities
(like concepts, topics, or facts) are nodes, and relationships between them are edges.
It helps systems understand connections between pieces of information rather than
treating them as isolated facts.
"""

# Step 4: LLM ko request bhejo
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful research assistant. Use the reference text to answer accurately."
        },
        {
            "role": "user",
            "content": f"Reference text:\n{reference_text}\n\nQuestion: {question}"
        }
    ],
    temperature=0.3,
)

# Step 5: Response print karo
print("LLM Response:\n")
print(response.choices[0].message.content)
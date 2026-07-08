# Intro to LLM

## Resources
- video 1 : https://www.youtube.com/watch?v=zjkBMFhNj_g
- video 2 : https://www.youtube.com/watch?v=xZDB1naRUlk

## Video 1 : Intro to Large Language Models

### What I Learned

- Large Language Models (LLMs) are neural networks trained to predict the next token in a sequence.
- They are built using the Transformer architecture, where self-attention helps understand context and relationships between words.
- Models are first pretrained on massive amounts of text and later refined using instruction tuning and RLHF to make them more helpful and conversational.
- LLMs don't "know" facts like a database—they generate text based on learned patterns, which is why they can sometimes hallucinate.
- Prompt quality plays an important role in the quality of the model's responses.
- common LLM security challenges,
    - Jailbreaks: prompts designed to bypass safety rules.
    - Prompt Injection: malicious instructions hidden in user inputs or external data to manipulate the model.
    - Data Poisoning: corrupting training data to influence the model's behavior or outputs.

---

## Video 2 : Development with Large Language Models Tutorial – OpenAI, Langchain, Agents, Chroma 
LLM = Large Language Model
• Large neural network trained on huge text datasets
• Learns by predicting the next token

### TRAINING PIPELINE

Raw Text
    ↓
Tokenization
    ↓
Neural Network Training
    ↓
Next Token Prediction
    ↓
Fine Tuning
    ↓
RLHF
    ↓
ChatGPT

### TOKENS

Token = Unit of text

Examples

"I love Python"

↓

["I","love","Python"]

↓

IDs

[345,821,1956]

• Uses Stop Token to end generation.

### NEURAL NETWORK

• Contains billions of parameters
• Learns language patterns
• Training:
    Input
      ↓
    Prediction
      ↓
    Compare with correct answer
      ↓
    Update parameters

### FINE TUNING

Purpose:
• Teach model new behaviour
• Domain-specific tasks
• Smaller labelled dataset
• Medical chatbot
• Customer support
• For adding knowledge, RAG is often preferred.

### RLHF

RLHF = Reinforcement Learning from Human Feedback

Purpose:
• Safer responses
• Reduce harmful outputs
• Align model with human preferences

### PROMPTING
Prompt = Input to LLM
Completion = Model response

### INFERENCE PARAMETERS

Temperature
• Creativity
• Low → deterministic
• High → creative

Top-p
• Limits candidate tokens

Frequency Penalty
• Reduces repeated words

Presence Penalty
• Encourages new topics

### SYSTEM MESSAGE
Defines model behaviour.
Example
"You are a helpful coding assistant."
"You are Shakespeare."

### FEW-SHOT PROMPTING
Give examples before asking question.
Example :
Input
Output
Input
Output
Input
?
Model learns expected format.

### PROMPT OPTIMIZATION
Use GPT to improve prompts.
Goal:
• Smaller prompts
• Better answers
• Lower token cost
• Tokens = pieces of words
• More tokens = Higher API cost.


### CHAINLIT

Framework for building ChatGPT-like interfaces.

User
    ↓
on_message()
    ↓
Call LLM
    ↓
Return Response

### LANGCHAIN

Framework for LLM applications.
Main Components :

#### PROMPT TEMPLATE
Reusable prompt.
Template
Answer this:
{question}
↓
question="What is AI?"

### LLMCHAIN
Combines Prompt + LLM
Produces output.

### STREAMING
Instead of waiting, tokens appear one by one.

### EMBEDDINGS
Embedding = Numerical vector representing meaning.
Similar meanings
↓
Nearby vectors.

### VECTOR DATABASE
Stores embeddings.

Examples
• ChromaDB
• Pinecone
• FAISS

Purpose

Fast similarity search.

### RAG (Retrieval Augmented Generation)
Document
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector DB
    ↓
Retriever
    ↓
LLM
    ↓
Answer

### CHUNKING

Large document
↓
Split into smaller chunks
↓
Generate embeddings
↓
Store in Vector DB

### COSINE SIMILARITY

Measures similarity between embeddings.
Lower distance = More similar.

### AGENTS
Agent = LLM that can use tools.
Workflow
Question
↓
Think
↓
Choose Tool
↓
Observe
↓
Think Again
↓
Answer

### TOOLS
• Google Search
• Arxiv
• Calculator
• Human Input
• Python
#### ARXIV TOOL
Used for

• Research papers
• Scientific search
• Latest publications
#### HUMAN AS A TOOL
Agent asks user when it needs additional information.
Useful for
• Clarification
• Verification
• Interactive workflows


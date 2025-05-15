import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline  # Updated import
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableSequence  # Use RunnableSequence instead of LLMChain

# ========== 1. Load Phi-2 ==========
model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=None)
model = AutoModelForCausalLM.from_pretrained(model_id, token=None, torch_dtype="auto", device_map="auto")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

model_llm = HuggingFacePipeline(pipeline=pipe)

# ========== 2. Define Few-Shot Examples ==========
examples = [
    {"input": "What are credit hours?", "output": "Credit hours represent the number of hours you spend in class per week. For instance, a 3-credit course typically means 3 hours of class time weekly."},
    {"input": "How many credit hours are needed to graduate?", "output": "Undergraduate students typically need 120 credit hours to graduate, though it depends on the program."},
    {"input": "What are course prerequisites?", "output": "Prerequisites are courses you must complete before enrolling in a more advanced class."},
    {"input": "Can you explain how to choose my major?", "output": "Choosing your major is an important decision that should reflect your interests and career goals. Meet with your academic advisor to explore your options."},
    {"input": "What are the general education requirements?", "output": "General education requirements are courses that all students must complete, regardless of their major, to ensure a well-rounded education."},
    {"input": "How does grading work?", "output": "Grading typically involves assignments, exams, and participation. Each course will have its own grading criteria, which should be outlined by your professor."}
]

# ========== 3. Memory ==========
memory = ConversationBufferMemory(memory_key="history", input_key="input")

# ========== 4. Prompt with Context and Examples ==========
prompt = PromptTemplate(
    input_variables=["history", "input", "examples"],
    template="""
You are an AI Academic Advisor at AUST. You ONLY respond to academic-related questions like courses, CGPA, credit hours, faculty, counseling, and registration. Keep replies short and clear.

Conversation History:
{history}

Here are some common questions and answers:

{examples}

Student: {input}
Advisor:"""
)

# ========== 5. Chain using RunnableSequence ==========
qa_chain = RunnableSequence(prompt | model_llm).with_config(memory=memory)  # Using RunnableSequence instead of LLMChain

# ========== 6. Run Chat with Memory ==========
if __name__ == "__main__":
    print("\nAUST Academic Advisor Chatbot (With Memory)")
    print("Ask your academic questions. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        
        # Exit condition
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        if not user_input:
            continue

        # Get the response from Phi-2 with the context and examples
        response = qa_chain.invoke({"input": user_input, "examples": examples}).strip()  # Use invoke for the new chain method
        print("Bot:", response)

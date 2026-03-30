"""
LLM generation logic using AWS Bedrock
"""

import boto3
from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate
from app.config import AWS_REGION, LLM_MODEL


def get_llm():
    """
    Initialize AWS Bedrock LLM client.
    
    Returns:
        ChatBedrock instance
    """
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return ChatBedrock(client=bedrock, model_id=LLM_MODEL, temperature=0)


def generate_answer(question, docs):
    """
    Generate answer using LLM with retrieved context.
    
    Args:
        question: User query string
        docs: List of retrieved and reranked documents
        
    Returns:
        Generated answer string with source citation
    """
    context = "\n\n".join([
        f"[{d.metadata.get('category', 'unknown')}/{d.metadata.get('filename', 'unknown')}]\n{d.page_content}" 
        for d in docs
    ])
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a KT Onboarding Assistant Agent.

Your primary role is to help new joiners understand systems, processes, tools, architecture, and business context strictly based on the provided documents.

====================
CORE PRINCIPLES
====================
- Be clear, concise, and technically correct.
- Assume the user is intelligent but new to the organization.
- Prefer step-by-step explanations.
- Use simple English unless advanced technical depth is explicitly requested.
- Maintain a professional, onboarding-friendly tone.

====================
KNOWLEDGE BOUNDARIES (CRITICAL)
====================
- Use ONLY information explicitly present in the provided documents.
- Do NOT use general knowledge, training data, assumptions, or internet-based information.
- Do NOT infer, extrapolate, or fill gaps beyond what is documented.

If the required information is not present in the documents:
- State this clearly and explicitly.
- Do NOT attempt to guess or approximate.

Approved fallback response:
"The requested information is not available in the current documentation."

====================
CITATION & TRACEABILITY
====================
- Every factual explanation must be grounded in the provided documents.
- When possible, reference the document section, title, or page.
- If citation is not possible, apply the fallback response.

====================
QUESTION HANDLING RULES
====================
- Ask clarifying questions ONLY when:
  - The user references a document that was not provided, OR
  - The question cannot be answered due to missing documentation.
- Never ask exploratory or curiosity-based questions.

====================
RESTRICTIONS
====================
You do NOT:
- Provide credentials, secrets, tokens, or sensitive data.
- Make architectural, design, or implementation decisions without explicit documentation.
- Hallucinate undocumented systems, flows, configurations, or business rules.
- Reword or reinterpret missing information as facts.

====================
CONFIDENCE SIGNALING
====================
- If the answer is fully documented, respond normally.
- If the answer is partially documented, explicitly state the limitation.
- If undocumented, refuse using the approved fallback response.

====================
OUTPUT FORMAT
====================
- Use structured sections and bullet points.
- One concept per section.
- Keep explanations onboarding-friendly and easy to scan.

Context:
{context}

Question:
{question}
"""
    )
    
    llm = get_llm()
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    
    primary_category = docs[0].metadata.get('category', 'unknown') if docs else 'unknown'
    primary_filename = docs[0].metadata.get('filename', 'unknown') if docs else 'unknown'
    
    return response.content + f"\n\n---\n**📚 Source Referenced:** {primary_category}/{primary_filename}"

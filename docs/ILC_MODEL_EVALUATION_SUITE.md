# Indian Legal Corpus (ILC) Evaluation Suite
## Framework for Testing Llama 3.1 Fine-Tuned Models

To truly evaluate your fine-tuned model (`Amarthya11/Llama3.1-finetuned-GGUF`), you cannot just ask it general questions. You need to test it on the specific dimensions where legal LLMs fail: **Hallucination**, **Reasoning**, **Context Adherence**, and **Format Compliance**.

This evaluation suite is designed to be used in two ways:
1. **Closed-Book Test**: Ask the model without providing RAG context (tests what the model learned during fine-tuning on the ILC).
2. **RAG-Augmented Test**: Provide a chunk of text and ask the question (tests the model's ability to strictly follow the context without hallucinating).

---

## Part 1: Legal Reasoning & Synthesis
*Tests if the model understands Indian legal doctrines and can synthesize an answer logically.*

**Q1. The Basic Structure Doctrine**
> **Prompt**: "Explain the evolution of the 'Basic Structure Doctrine' in Indian Constitutional Law. Cite the landmark judgments that established and reaffirmed this doctrine."
> **What to Evaluate**: The model *must* mention `Kesavananda Bharati v. State of Kerala (1973)`. Bonus points if it mentions `Minerva Mills (1980)` or `I.R. Coelho (2007)`. It should accurately describe that Parliament cannot alter the basic features of the Constitution.

**Q2. Conflict of Laws (Article 14 vs. Article 21)**
> **Prompt**: "Under what circumstances can a state action that passes the test of Article 14 (Right to Equality) still be struck down under Article 21 (Right to Life and Personal Liberty)? Provide an analysis based on Supreme Court precedents."
> **What to Evaluate**: Look for mentions of the *Maneka Gandhi v. Union of India (1978)* case. The model should explain the "just, fair, and reasonable" standard.

**Q3. Statutory Interpretation**
> **Prompt**: "What are the essential ingredients required to constitute an offence of Cheating under Section 415 of the Indian Penal Code (IPC)?"
> **What to Evaluate**: Accuracy. It must explicitly state the ingredients: (1) Deception of any person, (2) Fraudulently or dishonestly inducing that person to deliver any property, or (3) Intentional inducement to do or omit to do anything which causes damage or harm.

---

## Part 2: The Hallucination Trap (Adversarial Testing)
*Tests if the model is brave enough to say "I don't know" or if it invents fake Indian case law (The NCLT problem).*

**Q4. The Fake Precedent Trap**
> **Prompt**: "Summarize the Supreme Court's ruling in *State of Maharashtra v. Ritesh Varma (2025)* regarding the unconstitutionality of algorithmic surveillance."
> **What to Evaluate**: **CRITICAL PASS/FAIL.** This is a fake case. The model *must* state that it cannot find this case or that it does not exist. If it generates a summary, your model is hallucinating and failing the NCLT test.

**Q5. The Misattributed Quote**
> **Prompt**: "In which judgment did Justice D.Y. Chandrachud famously state that 'Liberty is not a gift of the State, but the very essence of human existence'?"
> **What to Evaluate**: If the model confidently attributes this to a specific case (like *Puttaswamy* or *Navtej Singh Johar*), you need to verify it. Often, LLMs will confidently invent a case for a dramatic quote.

---

## Part 3: Context Adherence (RAG Stress Test)
*Tests if your model can strictly follow the provided text without relying on its internal weights.*

**Q6. Contradiction Test**
> **Prompt**: "Based ONLY on the following context, what is the maximum punishment for defamation? 
> *Context: Under the newly amended Bharatiya Nyaya Sanhita (BNS) 2023, the maximum punishment for simple defamation has been reduced to a fine of Rs. 1,000 with no imprisonment.*" (Note: This context is factually incorrect on purpose).
> **What to Evaluate**: The model must output "A fine of Rs. 1,000 with no imprisonment." If it says "2 years imprisonment" (which is the actual law under IPC 499/500), it has failed the RAG test because it ignored your context and used its internal weights.

---

## Part 4: Formatting & Output Structuring
*Tests if the model can follow the strict output templates you require for your Streamlit UI.*

**Q7. The JSON Tool-Call Test**
> **Prompt**: "I need a full, comprehensive summary of the primary case. My Parent ID is 'SC-1994-Bommai'. Reply exactly with the JSON tool call required to retrieve the full document."
> **What to Evaluate**: Does it output exactly `{"tool_call": "retrieve_full_document", "source_id": "SC-1994-Bommai"}` without any conversational fluff (like "Sure, here is the JSON...")?

**Q8. The Digest Template Test**
> **Prompt**: "Summarize the provided context into exactly 3 sections: 1. Factual Matrix, 2. Ratio Decidendi, 3. Operative Order."
> **What to Evaluate**: Does it strictly output only those 3 sections? Are they correctly formatted with markdown headers?

---

### How to Score Your Model
When you test the GGUF model in your Hugging Face Space, score each answer on a scale of 0 to 3:
*   **0**: Complete hallucination or failure to follow instructions.
*   **1**: Partially correct, but includes conversational fluff or minor factual errors.
*   **2**: Factually correct and follows instructions, but reasoning is shallow.
*   **3**: Perfect answer. Legally sound, strictly follows constraints, identifies fake cases.

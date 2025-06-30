 #(Part 3: Agentic Workflow for Threat Analysis)

import os
import json
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS # Use langchain_community for specific integrations
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm

# --- Configuration ---
MODEL_NAME = "microsoft/phi-2" # Must match the base model used for fine-tuning
ADAPTERS_OUTPUT_DIR = "models/fine_tuned_llm_adapters"
VECTOR_STORE_DIR = "vector_store"
EMBEDDING_MODEL_NAME = "sentence-transformers/bge-small-en"

# QLoRA configuration for loading (must match training)
NF4_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 # Use bfloat16 for better performance on modern CPUs if supported, otherwise float16/float32
)

class AgenticThreatAnalyst:
    def __init__(self):
        print("Initializing Agentic Threat Analyst...")
        self.tokenizer = None
        self.model = None
        self.embedding_model = None
        self.vectorstore = None

        self._load_models()
        self._load_vector_store()
        
        # Define basic prompt templates for agents
        self.query_decomposer_prompt = PromptTemplate.from_template(
            """As a cybersecurity expert, analyze the user's query and identify key entities (e.g., CVE ID, threat actor, attack technique, malware name) and the user's core intent (e.g., summarize, detect, mitigate, analyze). List them as a JSON object with 'entities' (list of strings) and 'intent' (string).
            
            User Query: {query}
            
            JSON Output:"""
        )

        self.synthesis_analysis_prompt = PromptTemplate.from_template(
            """Based on the following cybersecurity context and the user's original query, provide a concise and actionable analysis. Focus on key findings, potential impacts, and practical recommendations. Use cybersecurity terminology accurately.
            
            Original User Query: {original_query}
            
            Retrieved Context:
            {context}
            
            Analysis and Recommendations:"""
        )

        self.output_action_prompt = PromptTemplate.from_template(
            """Based on the cybersecurity analysis, summarize the key takeaways and suggest concrete next steps for a security analyst.

            Cybersecurity Analysis: {analysis}

            Summary and Next Steps:"""
        )

    def _load_models(self):
        """Loads the fine-tuned LLM and embedding model."""
        try:
            print(f"Loading tokenizer for {MODEL_NAME}...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"Loading base LLM {MODEL_NAME} with 4-bit quantization for CPU inference...")
            # Load the base model with quantization config
            base_model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=NF4_CONFIG,
                trust_remote_code=True,
                device_map="cpu" # Explicitly load to CPU for free tier VM
            )
            # Load LoRA adapters and merge them for inference
            print(f"Loading and merging LoRA adapters from {ADAPTERS_OUTPUT_DIR}...")
            self.model = PeftModel.from_pretrained(base_model, ADAPTERS_OUTPUT_DIR)
            # For CPU, it's often better to keep LoRA adapters separate if memory is extremely tight
            # and only merge if you have enough RAM. For Phi-2/3, merging might be feasible.
            # self.model = self.model.merge_and_unload() # Uncomment if you want to merge

            print(f"Loading embedding model {EMBEDDING_MODEL_NAME}...")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print("Models loaded successfully.")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Ensure models/fine_tuned_llm_adapters and the base model are correctly downloaded/configured.")
            exit()

    def _load_vector_store(self):
        """Loads the FAISS vector store."""
        try:
            faiss_index_path = os.path.join(VECTOR_STORE_DIR, "faiss_index.bin")
            faiss_metadata_path = os.path.join(VECTOR_STORE_DIR, "faiss_metadata.json")

            if not os.path.exists(faiss_index_path) or not os.path.exists(faiss_metadata_path):
                print(f"Error: FAISS index or metadata not found. Run 'preprocess_data.py' first.")
                exit()

            print(f"Loading FAISS index from {faiss_index_path}...")
            index = faiss.read_index(faiss_index_path)

            print(f"Loading FAISS metadata from {faiss_metadata_path}...")
            with open(faiss_metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Reconstruct documents for FAISS
            # LangChain's FAISS class expects a list of Document objects
            from langchain_core.documents import Document
            docs = []
            for i, m in enumerate(metadata):
                # Retrieve the actual text content that corresponds to this embedding
                # This assumes your preprocess_data.py saved the actual text chunk
                # alongside the metadata, or that you can retrieve it by index.
                # For simplicity here, we'll assume a dummy content for demonstration.
                # In a real setup, you'd save the text chunks with metadata.
                # For this simple FAISS usage, we need a content for Document.
                # Simplest for now is to store chunk_text in metadata too during preprocessing.
                # Or, load chunks here from a separate file saved by preprocess_data.py.
                # For free tier, saving chunks directly to metadata JSON is best to reduce files.
                docs.append(Document(page_content=m.get('content', f"Chunk {i}"), metadata=m))

            # Temporarily create a dummy embedding function for FAISS constructor, as we already have embeddings
            # LangChain FAISS is designed to embed itself, but we pre-embedded.
            # We'll use a wrapper to make it work.
            class DummyEmbeddings:
                def embed_documents(self, texts):
                    return self.embedding_model.encode(texts).tolist()
                def embed_query(self, text):
                    return self.embedding_model.encode([text])[0].tolist()

            # The current FAISS.from_texts or .from_documents re-embeds.
            # We need to use FAISS directly or a custom wrapper for pre-computed embeddings.
            # For simplicity, we'll simulate LangChain's retriever capabilities directly here.
            self.faiss_index = index
            self.faiss_metadata = metadata
            print("Vector store loaded successfully.")
        except Exception as e:
            print(f"Error loading vector store: {e}")
            print("Ensure 'vector_store/' contains 'faiss_index.bin' and 'faiss_metadata.json'.")
            exit()

    def _get_llm_chain(self, prompt_template):
        """Helper to create a LangChain LLM chain with the fine-tuned model."""
        # Wrap the Hugging Face model for LangChain
        class CustomHFLLM(torch.nn.Module): # Inherit from nn.Module to use as callable
            def __init__(self, model, tokenizer):
                super().__init__()
                self.model = model
                self.tokenizer = tokenizer
                self.tokenizer.pad_token = self.tokenizer.eos_token # Ensure padding token is set

            def __call__(self, prompt: str, **kwargs) -> str:
                # Prepare inputs
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    return_attention_mask=True,
                    padding=True, # Pad to max_length for batch inference if any, else no padding
                    truncation=True, # Truncate if too long
                    max_length=512 # Keep max_length low for free tier
                ).to(self.model.device) # Ensure inputs are on the correct device (CPU)

                # Generate output
                # Use generate with appropriate parameters for CPU
                # num_beams=1, do_sample=False for deterministic, faster generation
                # max_new_tokens=100-200 for concise answers
                output_sequences = self.model.generate(
                    **inputs,
                    max_new_tokens=256, # Keep output short for free tier
                    do_sample=False, # Deterministic generation is faster
                    num_beams=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                generated_text = self.tokenizer.decode(output_sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                return generated_text.strip() # Remove leading/trailing whitespace

        llm_callable = CustomHFLLM(self.model, self.tokenizer)
        return LLMChain(prompt=prompt_template, llm=llm_callable, output_parser=StrOutputParser())


    def retrieve_info(self, query: str, k=3):
        """Retrieves relevant documents from the FAISS vector store."""
        if not self.faiss_index or not self.embedding_model:
            print("Vector store or embedding model not loaded.")
            return []

        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        retrieved_docs = []
        for i, idx in enumerate(indices[0]):
            metadata_item = self.faiss_metadata[idx]
            # Assuming 'content' key in metadata directly holds the chunk text for simplicity
            retrieved_docs.append(f"Source: {metadata_item.get('source', 'N/A')}, ID: {metadata_item.get('id', metadata_item.get('filename', 'N/A'))}\nContent:\n{metadata_item.get('content', 'Content not available.')}")
        return retrieved_docs

    def analyze_threat(self, user_query: str):
        """Orchestrates the agentic workflow."""
        print(f"\n--- Processing Query: '{user_query}' ---")

        # Agent 1: Query & Intent Decomposer
        print("Agent 1: Decomposing query...")
        query_decomposer_chain = self._get_llm_chain(self.query_decomposer_prompt)
        try:
            decomposed_output_str = query_decomposer_chain.run(query=user_query)
            decomposed_output = json.loads(decomposed_output_str) # Expects JSON output
            entities = decomposed_output.get("entities", [])
            intent = decomposed_output.get("intent", "general_inquiry")
            print(f"Decomposed: Entities={entities}, Intent={intent}")
        except json.JSONDecodeError:
            print(f"Warning: LLM did not return valid JSON for decomposition. Raw: {decomposed_output_str}")
            entities = [user_query] # Fallback to original query
            intent = "general_inquiry"
        except Exception as e:
            print(f"Error in Agent 1: {e}. Falling back to basic entities.")
            entities = [user_query]
            intent = "general_inquiry"


        # Agent 2: Multi-Source Retrieval Agent
        print("Agent 2: Retrieving information...")
        retrieval_queries = entities if entities else [user_query]
        all_retrieved_context = []
        for q in tqdm(retrieval_queries, desc="Retrieving for entities"):
            all_retrieved_context.extend(self.retrieve_info(q, k=2)) # Retrieve fewer docs for free tier

        if not all_retrieved_context:
            return "No relevant information found in the knowledge base for your query. The cyber landscape shifts daily, and this prototype has a limited static dataset."
        
        # Combine and deduplicate retrieved context
        unique_context = "\n\n---\n\n".join(sorted(list(set(all_retrieved_context))))
        print(f"Retrieved {len(all_retrieved_context)} context chunks.")

        # Agent 3: Synthesis & Analysis Agent
        print("Agent 3: Synthesizing and analyzing...")
        synthesis_analysis_chain = self._get_llm_chain(self.synthesis_analysis_prompt)
        analysis = synthesis_analysis_chain.run(original_query=user_query, context=unique_context)
        print("Analysis complete.")

        # Agent 4: Output & Action Suggestion Agent
        print("Agent 4: Generating summary and next steps...")
        output_action_chain = self._get_llm_chain(self.output_action_prompt)
        final_output = output_action_chain.run(analysis=analysis)
        print("Output generation complete.")

        return final_output

if __name__ == "__main__":
    analyst = AgenticThreatAnalyst()

    # --- Example Queries ---
    queries = [
        "What are the latest details on CVE-2023-0001 and its mitigation?",
        "Summarize recent activity of APT29 and recommend detection methods.",
        "Explain the typical stages of a spearphishing attack (T1566.001) and how to protect against it.",
        "What is a zero-day vulnerability and how critical is it?"
    ]

    for query in queries:
        response = analyst.analyze_threat(query)
        print("\n" + "="*50)
        print("Final Response to Analyst:")
        print(response)
        print("="*50 + "\n")

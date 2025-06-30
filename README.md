# Agentic Threat Intelligence Analyst for Emerging Cyber Threats
### agentic_cyber_analyst

## Project Overview
This project implements a miniature prototype of an "Agentic Threat Intelligence Analyst," an AI system designed to assist cybersecurity professionals by proactively gathering, synthesizing, and recommending actions in response to emerging cyber threats. It specifically explores how a combination of agentic AI principles, retrieval-augmented generation (RAG), and domain-specific fine-tuning can overcome the limitations of traditional rule-based systems and generic Large Language Models (LLMs) in the dynamic and specialized cybersecurity landscape.

Free Tier Optimization

This project has been deliberately optimized for deployment within the Google Cloud Platform (GCP) Free Tier. This design choice necessitated extreme selectivity in data volume, careful model selection, and efficient processing strategies to demonstrate the core concepts within strict resource constraints (primarily CPU-only VMs for inference and limited GPU access for fine-tuning via Google Colab).

## Architecture
The system operates through a multi-stage, agentic RAG pipeline:

- Foundational RAG System: A small, curated knowledge base of cybersecurity threat intelligence (CVEs, MITRE ATT&CK, advisories, glossary) is preprocessed, chunked, and embedded into a local FAISS vector database.

- Domain-Specific Fine-tuning: A small, open-source LLM (e.g., Phi-2/3 Mini) is fine-tuned using Parameter-Efficient Fine-Tuning (PEFT) techniques on a highly curated, small dataset of cybersecurity-specific instruction-response pairs. This enhances its understanding of technical jargon and threat intelligence analysis patterns.

- Agentic Workflow: An orchestration layer (using LangChain) manages a sequence of specialized agents:
 - Query & Intent Decomposer: Parses the user's natural language query using the fine-tuned LLM to extract key entities and intent.
 - Multi-Source Retrieval Agent: Performs targeted semantic searches against the vector database based on the decomposed query, retrieving relevant context.
 - Synthesis & Analysis Agent: Leverages the fine-tuned LLM to synthesize retrieved information, infer relationships, summarize findings, and generate actionable recommendations.
 - Output & Action Suggestion Agent: Formats the analysis and proposes concrete next steps for a human security analyst

## Setup and Installation

#### Prerequisites

Python 3.9+

Access to a Google Cloud Platform account (Free Tier).

Google Colab (for fine-tuning, leveraging free GPU access).

#### Steps

1. Clone the Repository
   ```
   git clone https://github.com/your-username/agentic-cyber-analyst.git
   cd agentic-cyber-analyst
   ```
2.  Install Dependencies

It's recommended to create a virtual environment:

```
Bash
python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`
pip install -r requirements.txt
```


3. Prepare Your Data (Part 1 - RAG Knowledge Base)

Create the data/ directory and populate it with your highly curated, small datasets:

- data/cves.json (e.g., 5-10 relevant CVE entries)
- data/mitre_attack.json (e.g., 3-5 Tactics, 10-15 Techniques)
- data/advisories/ (folder for 1-2 small plain text or HTML advisories)
- data/glossary.txt (e.g., 5-10 core cybersecurity terms and definitions)

Example Data Structures (see preprocess_data.py for exact formats):

- cves.json: [{"id": "CVE-XXXX-YYYY", "description": "...", "severity": "..."}]
- mitre_attack.json: [{"id": "TXXXX", "name": "...", "description": "..."}]
- glossary.txt: Plain text definitions.

After populating, run the preprocessing script to build your FAISS vector store:

```
python preprocess_data.py
```

4. Fine-tune the LLM (Part 2 - Domain-Specific Understanding)

This step is best performed in a Google Colab notebook due to GPU requirements, even with PEFT.

- Create your fine-tuning dataset: Populate finetuning_data/cyber_qa_pairs.jsonl with your very small (e.g., 50-100 high-quality) instruction-response pairs in JSONL format.

Example entry: {"instruction": "Explain MITRE ATT&CK T1566.001.", "output": "MITRE ATT&CK T1566.001 refers to 'Spearphishing Attachment'. ..."}

- Open finetune_llm.py in a Colab notebook.

Ensure GPU Runtime: Go to Runtime > Change runtime type and select T4 GPU (or similar).

- Upload your finetuning_data/ folder to the Colab environment.

- Run the script cells to install dependencies, load the base LLM (e.g., microsoft/phi-2), apply LoRA, and train.

- Upon successful completion, the fine-tuned LoRA adapters will be saved to ./models/fine_tuned_llm_adapters. Download this fine_tuned_llm_adapters folder to your local machine (and later upload to your GCP VM).

5. Deploy and Run the Agentic System (Part 3 - GCP Free Tier VM)

- Set up a Google Cloud Free Tier VM:

- Create a new Compute Engine VM instance.

- Select a machine type eligible for the free tier (e.g., e2-micro or e2-small).

- Choose an Ubuntu or Debian operating system.

- Allow HTTP/HTTPS traffic (though not strictly needed for backend processing, good practice).

- Crucially, ensure you do NOT select a GPU. This project is designed for CPU-only inference on the free tier.

- SSH into your VM and clone your repository.
- 
Transfer the fine_tuned_llm_adapters folder (downloaded from Colab) into the models/ directory of your cloned repository on the VM.

- Install dependencies on the VM as in step 2.

Run the main agentic analysis script:

```
python main.py
```

The script will load the models, vector store, and process predefined example queries. You can modify main.py to accept interactive input if desired.

### Possible Future Work

- Expand Knowledge Base & Live Feeds: Integrate with real-time threat intelligence feeds (e.g., STIX/TAXII feeds, OSINT sources) and larger public vulnerability databases for more up-to-date information.

- Enhance Agentic Capabilities: Implement more sophisticated tool-use integrations, such as querying external APIs for IP reputation, analyzing malware sandbox results, or interacting with SOAR (Security Orchestration, Automation, and Response) platforms.

- Reinforcement Learning from Human Feedback (RLHF): Further align the model's outputs with actual security analyst preferences and workflows for even more actionable and relevant recommendations.

- Scalable Deployment: Transition to paid GCP services (e.g., Vertex AI for model hosting and inference, Cloud Run with GPU, managed vector databases like AlloyDB for PostgreSQL with pgvector) for production-grade performance and scalability.

- Safety & Trustworthiness: Implement robust guardrails to prevent hallucinations, ensure data privacy, handle sensitive cybersecurity information responsibly, and address potential adversarial attacks against the LLM itself.






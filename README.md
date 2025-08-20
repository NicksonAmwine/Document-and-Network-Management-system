# Enterprise RAG System for Document and Network-log management
## Project Overview
This project is a **Retrieval-Augmented Generation (RAG)** system designed specifically for networking companies that deal with large volumes of technical documentation, logs, and alerts.

The goal is to help engineers and support teams quickly find accurate information from internal documents and network monitoring data, through a simple chat-like web interface.

Instead of manually digging through multiple PDFs or logs, users can just ask a question, and the system will retrieve the most relevant content, summarize it (if needed), and display the answer instantly.

## Key Features
**1. Document Search:** Ask natural questions and get answers from your internal documents instantly.

**2. Local LLM and embedding model Integration:** Uses a locally hosted open-source Large Language Model and embedding model for answering queries.

**3. Private Vector Database:** Embeddings and document vectors are stored in a local Chroma DB for maximum privacy.

**4. Network Log Analysis:** Future integration with network logs and alarm systems to assist with troubleshooting.

**5. Web Chat Interface:** Built with Django — allows employees to interact with the system via a browser-based chat UI.

**6. Dockerized:** Easy to deploy and scale using Docker containers.

**7. Runs 100% Locally:** All components run on internal servers for maximum data security.

## Use Cases
**1. Internal Document Q&A:** Quickly find config rules, setup procedures, internal architecture docs, and SOPs.

**2. Alarm & Traffic Log Analysis:** Help NOC/engineering teams troubleshoot escalated tickets using LLM-based insight from alarm logs and traffic trends.

## How It Works

**Document Ingestion:** Load internal docs (PDFs, etc.) from a shared drive (like Google Drive).

**Chunking + Embedding:** Documents are split into chunks and converted into vector embeddings.

**Vector Storage:** Embeddings are stored in a local Chroma DB instance.

**RAG Pipeline:** User asks a question → system retrieves relevant chunks → passes them to LLaMA → response generated.

**Chat Interface:** Users interact with the system via a Django-powered chat UI.

## Future Enhancements

- Integrate real-time alarm logs from monitoring tools like Zabbix, Nagios, or Prometheus.

- Add user authentication and access control.

- Track query history and feedback to improve retrieval quality.

## Intended Users

- Network Engineers

- Customer Support / NOC Analysts

- DevOps Teams

- Knowledge Management Staff

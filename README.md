# NyayaMitra (न्यायमित्र)

**AI-Powered Legal Assistant for Indian Citizens**

NyayaMitra ("Friend of Justice") is an AI legal assistant that provides accurate, cited, multilingual legal guidance covering Indian statutory law, judicial precedents, and step-by-step procedural walkthroughs.

> **Disclaimer:** This system provides legal information, not legal advice. For case-specific advice, consult a qualified advocate.

---

## Features

- **7 Legal Domains** — Criminal, Property, Family, Labor, Consumer, Constitutional, IP
- **Citation-Verified Responses** — Every section number and case name verified against the database
- **Hybrid Retrieval** — Semantic search (Qdrant) + Keyword search (Elasticsearch) + Knowledge Graph (Neo4j)
- **Structured Responses** — Applicable law, precedents, step-by-step procedure, jurisdiction notes
- **Multilingual** — English, Hindi, Tamil, Telugu, Bengali, Marathi (more coming)
- **Fine-Tuned LLM** — Llama 3.1 70B trained on Indian legal corpus with domain-specific LoRA adapters

---

## Tech Stack

| Layer           | Technology                                       |
| --------------- | ------------------------------------------------ |
| LLM             | Llama 3.1 70B (fine-tuned) / 8B (development)    |
| Inference       | vLLM                                             |
| Vector DB       | Qdrant                                           |
| Search          | Elasticsearch (BM25)                             |
| Knowledge Graph | Neo4j                                            |
| Embeddings      | BGE-large-en-v1.5 (fine-tuned)                   |
| Backend         | FastAPI (Python 3.11+)                           |
| Frontend        | Next.js 14+ (TypeScript, Tailwind CSS)           |
| Cache           | Redis                                            |
| Database        | PostgreSQL                                       |
| Orchestration   | Docker Compose (local) / Kubernetes (production) |

---

## Quick Start

### Prerequisites

- **Python 3.11+** (3.13 works)
- **Node.js 18+**
- **Docker Desktop** (for infrastructure services)
- **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/nyayamitra.git
cd nyayamitra
```

### 2. Setup Environment

```bash
# Create .env from template
cp .env.example .env

# IMPORTANT: Edit .env and set hosts to localhost for local development
# POSTGRES_HOST=localhost
# REDIS_HOST=localhost
# QDRANT_HOST=localhost
# ELASTICSEARCH_HOST=localhost
# NEO4J_HOST=localhost
```

### 3. Start Infrastructure

```bash
# Pull images and start all services
docker compose up -d

# Verify everything is running
docker compose ps
```

This starts: PostgreSQL (5432), Redis (6379), Qdrant (6333), Elasticsearch (9200), Neo4j (7474/7687).

### 4. Install Python Dependencies

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 5. Start the Backend

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

### 6. Verify

```bash
# Root endpoint
curl http://localhost:8080

# Health check (all services)
curl http://localhost:8080/api/v1/health/detail | python3 -m json.tool

# Test a legal query
curl -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Can police arrest me without a warrant?", "jurisdiction": "Maharashtra", "domain_hint": "criminal"}' \
  | python3 -m json.tool
```

### 7. Explore the API

Open **http://localhost:8080/docs** in your browser for interactive Swagger documentation.

---

## Project Structure

```
nyayamitra/
├── backend/                    # FastAPI application
│   ├── app/
│   │   ├── main.py            # Application entry point
│   │   ├── config.py          # Centralized settings (reads .env)
│   │   ├── routers/
│   │   │   ├── health.py      # Health check endpoints
│   │   │   └── query.py       # Legal query endpoints
│   │   ├── services/          # Business logic (retrieval, LLM, verification)
│   │   └── models/
│   │       └── query.py       # Pydantic schemas
│   └── requirements.txt
│
├── frontend/                   # Next.js application
│   └── src/
│       ├── app/               # Next.js App Router
│       ├── components/        # React components
│       └── lib/               # Utilities and API client
│
├── data/                       # Data pipeline
│   ├── scrapers/              # Indian Kanoon, India Code, SCI scrapers
│   ├── processors/            # Act parser, judgment parser, chunker
│   ├── embeddings/            # Embedding model, indexer, graph loader
│   └── datasets/              # SFT, DPO, and evaluation datasets
│
├── models/                     # ML model training and serving
│   ├── training/              # CPT, SFT, DPO training scripts
│   ├── serving/               # vLLM server, adapter manager
│   ├── embeddings/            # Embedding and re-ranker training
│   └── router/                # Query classification model
│
├── evaluation/                 # Legal accuracy benchmarks
│   ├── benchmarks/            # Citation, jurisdiction, procedural accuracy
│   └── datasets/              # Gold-standard evaluation questions
│
├── deployment/                 # Production infrastructure
│   ├── terraform/             # Infrastructure as Code
│   ├── kubernetes/            # K8s manifests
│   └── helm/                  # Helm charts
│
├── docker-compose.yml          # Local infrastructure stack
├── Makefile                    # Development commands
├── .env.example                # Environment variable template
└── README.md                   # This file
```

---

## Makefile Commands

| Command         | Description                                         |
| --------------- | --------------------------------------------------- |
| `make setup`    | First-time setup: create .env, install dependencies |
| `make start`    | Start all Docker infrastructure services            |
| `make stop`     | Stop all services (preserves data)                  |
| `make restart`  | Restart all services                                |
| `make reset`    | Wipe ALL data and restart fresh                     |
| `make status`   | Show health of all services                         |
| `make logs`     | Follow logs from all services                       |
| `make backend`  | Start FastAPI backend with hot reload               |
| `make frontend` | Start Next.js development server                    |
| `make ingest`   | Run data ingestion pipeline                         |
| `make index`    | Run embedding and indexing pipeline                 |
| `make test`     | Run all tests                                       |
| `make lint`     | Run linters (ruff, black, eslint)                   |
| `make format`   | Auto-format all code                                |
| `make eval`     | Run legal evaluation benchmark                      |
| `make clean`    | Remove cache files                                  |

---

## API Endpoints

| Endpoint                | Method | Description                     |
| ----------------------- | ------ | ------------------------------- |
| `/`                     | GET    | Application info and disclaimer |
| `/docs`                 | GET    | Swagger UI (development only)   |
| `/api/v1/health`        | GET    | Quick health check              |
| `/api/v1/health/detail` | GET    | Detailed check of all services  |
| `/api/v1/query`         | POST   | Submit a legal query            |
| `/api/v1/query/stream`  | POST   | Streaming legal query (SSE)     |
| `/api/v1/feedback`      | POST   | Submit feedback on a response   |

### Query Request

```json
{
  "query": "Can police arrest me without a warrant?",
  "language": "en",
  "jurisdiction": "Maharashtra",
  "domain_hint": "criminal",
  "detail_level": "detailed"
}
```

### Query Response

```json
{
  "response_id": "uuid",
  "answer": "Plain language explanation...",
  "applicable_law": [
    { "act": "CrPC, 1973", "section": "41", "text": "...", "status": "active" }
  ],
  "precedents": [
    {
      "case": "D.K. Basu v. State of WB",
      "year": 1997,
      "court": "Supreme Court",
      "citation": "(1997) 1 SCC 416",
      "relevance": "..."
    }
  ],
  "procedure": [
    {
      "step": 1,
      "action": "...",
      "details": "...",
      "forms": [],
      "court": "..."
    }
  ],
  "jurisdiction_notes": "State-specific variations...",
  "confidence": "high",
  "disclaimer": "This is legal information, not legal advice...",
  "sources_verified": true
}
```

---

## Development Workflow

### Branch Naming

- `feature/sprint-X-description` — New features
- `fix/issue-description` — Bug fixes
- `refactor/component` — Code improvements

### Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add hybrid retrieval with RRF fusion
fix: correct jurisdiction detection for Maharashtra
docs: update API schema documentation
refactor: extract citation verification into service
test: add benchmark for section citation accuracy
chore: update Docker images to latest versions
```

### Pull Request Process

1. Create a feature branch from `main`
2. Write code following the coding conventions
3. Ensure `make lint` and `make test` pass
4. Create PR with description of changes
5. Get at least 1 code review
6. Merge after CI passes

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│  Query Router    │  ← DistilBERT classifier
│  (domain, juris) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│       Hybrid Retrieval Engine       │
│  ┌─────────┬────────┬────────────┐  │
│  │ Qdrant  │  Elas  │   Neo4j    │  │
│  │ (dense) │ (BM25) │  (graph)   │  │
│  └────┬────┴───┬────┴─────┬──────┘  │
│       └────────┼──────────┘         │
│           RRF Fusion                │
│         Cross-Encoder               │
│          Re-ranking                 │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────┐
│  Fine-Tuned LLM │  ← Llama 3.1 70B + Domain LoRA
│   (via vLLM)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Citation      │  ← Verify every section & case
│    Verifier      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Translation    │  ← IndicTrans2
│     Layer        │
└────────┬────────┘
         │
         ▼
    Response
```

---

## Legal Domains

| Domain         | Key Legislation                                    |
| -------------- | -------------------------------------------------- |
| Criminal       | IPC/BNS, CrPC/BNSS, Indian Evidence Act            |
| Property       | Transfer of Property Act, Registration Act, RERA   |
| Family         | Hindu Marriage Act, Special Marriage Act, DV Act   |
| Labor          | Industrial Disputes Act, Labor Codes 2020, POSH    |
| Consumer       | Consumer Protection Act 2019, E-commerce Rules     |
| Constitutional | Constitution of India, RTI Act, Election Laws      |
| IP             | Patents Act, Trademarks Act, Copyright Act, IT Act |

---

## Contributing

Please read our coding conventions in the Project Knowledge document before contributing. Key points:

- **Python:** Black formatting (line length 100), Ruff linting, mypy type checking
- **TypeScript:** Strict mode, Tailwind CSS, no `any` types
- **Tests:** Every feature needs unit tests; every sprint needs legal expert review
- **Legal data:** Use only real Indian legal sections and cases. Never fabricate citations.

---

## License

This project is proprietary. All rights reserved.

---

## Contact

For questions about NyayaMitra development, reach out to the project team.

---

_NyayaMitra — Making Justice Accessible to Every Indian Citizen_

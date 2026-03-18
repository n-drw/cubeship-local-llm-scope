# Logistics RAG — Project Scope & Architecture

## 1. Executive Summary

A **locally-hosted RAG system** for a logistics/shipping client, built around a
**Mixture-of-Experts (MoE)** agentic architecture with 6 specialist agents
("claws"), a semantic-search vector database, fine-tuned central reasoning model,
and a React/Next.js front-end with JWT-based authentication.

Current customer data footprint: **~30 million tokens**.

---

## 2. Architecture Overview

> See `architecture.dot` — render with:
> ```bash
> dot -Tpng architecture.dot -o architecture.png
> dot -Tsvg architecture.dot -o architecture.svg
> ```

### Layer Summary

| # | Layer | Key Technologies |
|---|-------|-----------------|
| 1 | **Client** | Next.js 14+, React, TailwindCSS |
| 2 | **Auth** | NextAuth.js or Keycloak, JWT, Redis token store, RBAC |
| 3 | **API / Orchestrator** | FastAPI (Python) or Express (Node), NemoClaw/OpenClaw runtime |
| 4 | **Mixture of Experts** | 6 specialist agents (see §3) |
| 5 | **Inference** | vLLM / TRT-LLM, Nemotron-3 Nano, Triton Inference Server |
| 6 | **Storage** | Milvus/Qdrant (vectors), MongoDB/S3 (docs), PostgreSQL (metadata) |
| 7 | **Ingestion Pipeline** | Unstructured.io / LlamaParse, semantic chunker, batch embedder |
| 8 | **OpenClaw/NemoClaw** | Skill registry, privacy router, NeMo Guardrails, OpenShell sandbox |
| 9 | **Observability** | OpenTelemetry, Prometheus/Grafana, ELK/Loki |

---

## 3. Mixture of Experts — The 6 Specialist Agents

| Expert | Domain | Primary Task |
|--------|--------|-------------|
| **Expert 1 — OCR Agent** | Document digitization | Extract text from scanned BOLs, invoices, packing lists |
| **Expert 2 — Object Detection** | Cargo inspection | Zero-shot detection of packages, damage, labels from photos |
| **Expert 3 — Multi-Modal Reasoning** | Visual Q&A / analysis | Answer questions about cargo images + shipping docs |
| **Expert 4 — Data Pipeline** | ETL | PDF → structured CSV/JSON → chunked embeddings |
| **Expert 5 — RAG / Retrieval** | Semantic search | Retrieve + rerank + generate answers from vector store |
| **Expert 6 — Analytics & Reporting** | Business intelligence | SQL generation, KPI dashboards, anomaly detection |

---

## 4. Model Comparison for MoE Experts

### 4.1 OCR Models

| Model | Params | Strengths | Weaknesses | License | Best For |
|-------|--------|-----------|------------|---------|----------|
| **GOT-OCR2.0** | 580M | End-to-end, multi-page, sheet music & math; strong on dense layouts | Higher VRAM (~4GB); slower than pipeline OCR | Apache 2.0 | Complex logistics docs with tables/stamps |
| **Surya** | ~200M | Fast, multilingual (90+ langs), line-level detection + recognition | Less accurate on heavily rotated/skewed scans | GPL-3.0 (⚠️) | High-volume multilingual BOLs |
| **DocTR** | ~30M | Lightweight, CPU-friendly, modular (detection + recognition) | Lower accuracy on handwritten text | Apache 2.0 | Edge/CPU-only deployments |
| **PaddleOCR (PP-OCRv4)** | ~12M | Ultra-light, mobile-ready, 80+ languages, strong CJK | Ecosystem is PaddlePaddle-centric | Apache 2.0 | High-throughput server-side OCR |
| **EasyOCR** | ~20M | Simple API, 80+ languages, good community | Slower than PaddleOCR; less accurate on dense tables | Apache 2.0 | Quick prototyping |

**Recommendation:** **GOT-OCR2.0** for accuracy on complex logistics documents (BOLs with stamps, barcodes, tables). Fall back to **PaddleOCR** for high-throughput batch ingestion where speed > accuracy.

---

### 4.2 Zero-Shot Object Detection Models

| Model | Params | Strengths | Weaknesses | License | Best For |
|-------|--------|-----------|------------|---------|----------|
| **OWLv2 (OWL-ViT v2)** | 140M–400M | Google; strong zero-shot text-conditioned detection; fast | Weaker on very small objects | Apache 2.0 | Open-vocab cargo label detection |
| **Grounding DINO** | 172M–340M | State-of-art open-set; phrase grounding; integrates with SAM | Heavier compute; complex pipeline | Apache 2.0 | Damage inspection with text prompts |
| **YOLO-World** | 17M–76M | Real-time; YOLO speed + open vocabulary; edge-friendly | Less precise than Grounding DINO on novel categories | GPL-3.0 (⚠️) | Real-time video/conveyor-belt detection |
| **Florence-2** | 230M–770M | Microsoft; unified vision model; detection + captioning + OCR | Larger footprint; overkill for detection-only | MIT | Combined detection + captioning tasks |

**Recommendation:** **Grounding DINO** for highest accuracy on novel cargo categories (damage, label reading). **YOLO-World** as a real-time fallback for warehouse conveyor-belt scenarios where latency < 50ms is required.

---

### 4.3 Multi-Modal (Vision-Language) Models

| Model | Total / Active Params | Strengths | Weaknesses | License | Best For |
|-------|----------------------|-----------|------------|---------|----------|
| **Qwen2.5-VL-7B** | 7B | Leading VLM at its size; video understanding; agentic tool use | Needs ~16GB VRAM (FP16) | Apache 2.0 | Primary multi-modal expert |
| **LLaVA-NeXT (LLaVA-OneVision)** | 7B–72B | Strong visual reasoning; multi-image; open weights | 7B variant slightly behind Qwen2.5-VL | Apache 2.0 | Complex multi-image comparisons |
| **Phi-3.5-Vision** | 4.2B | Very small; runs on 8GB VRAM; strong chart/doc understanding | Weaker on open-domain images | MIT | Resource-constrained deployments |
| **InternVL2.5** | 2B–78B | Top benchmark scores; dynamic resolution; strong OCR | Largest variants need multi-GPU | Apache 2.0 | When accuracy is paramount |
| **Moondream2** | 1.8B | Tiny; runs on edge/CPU; fast inference | Limited reasoning depth | Apache 2.0 | Edge cameras / IoT devices |

**Recommendation:** **Qwen2.5-VL-7B** as the primary multi-modal expert (best accuracy/resource ratio). **Phi-3.5-Vision** as the lightweight alternative for resource-constrained setups.

---

### 4.4 Central Reasoning / Orchestrator LLM

| Model | Total / Active Params | Architecture | Strengths | Best For |
|-------|----------------------|-------------|-----------|----------|
| **Nemotron-3 Nano** | 30B / 3.5B active | Hybrid Mamba2-Transformer MoE (128 experts, 6 active) | Best throughput-per-quality; agentic reasoning; open weights | Primary central model |
| **Qwen2.5-32B** | 32B | Dense Transformer | Strong reasoning, multilingual | Fallback / comparison |
| **DeepSeek-V3-Lite** | ~16B active | MoE Transformer | Cost-efficient MoE | Budget alternative |
| **Mistral Small 3.1 24B** | 24B | Dense Transformer | Fast, Apache 2.0, vision support | API-serving alternative |

**Recommendation:** **Nemotron-3 Nano 30B-A3B** — only 3.5B active parameters with 30B total, giving dense-model quality at a fraction of compute. Hybrid Mamba-Transformer MoE architecture is purpose-built for agentic reasoning and pairs natively with NemoClaw/OpenClaw.

---

### 4.5 Embedding & Reranking Models

| Model | Dims | Strengths | Best For |
|-------|------|-----------|----------|
| **NV-Embed-v2** | 4096 | NVIDIA; top MTEB scores; passage + instruction aware | Primary embedder |
| **BGE-M3** | 1024 | Multilingual; dense + sparse + ColBERT hybrid | Multilingual docs |
| **E5-Mistral-7B** | 4096 | LLM-based embeddings; strong on long context | Long documents |
| **NV-RerankQA-Mistral** | — | NVIDIA reranker; pairs with NV-Embed | Reranking pipeline |
| **BGE-Reranker-v2-M3** | — | Multilingual reranker | Multilingual reranking |

---

## 5. OpenClaw / NemoClaw Integration

**NVIDIA NemoClaw** is the enterprise-grade stack for OpenClaw that adds:

- **NVIDIA OpenShell** runtime — isolated sandbox for secure agent execution
- **Nemotron models** — install locally via a single command
- **Privacy Router** — policy-based routing between local and cloud models
- **NeMo Guardrails** — input/output filtering for safety and compliance
- **Always-on agents** — dedicated compute on RTX/DGX hardware

### 5.1 Proposed OpenClaw Workflow for This Project

```
┌──────────────────────────────────────────────────────────┐
│              NemoClaw Runtime (OpenShell Sandbox)         │
│                                                          │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐  │
│  │ SKILL: OCR  │   │ SKILL: ObjDet│   │ SKILL: VLM   │  │
│  │ SKILL.md    │   │ SKILL.md     │   │ SKILL.md     │  │
│  │ ─────────── │   │ ──────────── │   │ ──────────── │  │
│  │ •scan_doc() │   │ •detect()    │   │ •reason()    │  │
│  │ •extract()  │   │ •classify()  │   │ •caption()   │  │
│  │ •validate() │   │ •annotate()  │   │ •compare()   │  │
│  └──────┬──────┘   └──────┬───────┘   └──────┬───────┘  │
│         │                 │                   │          │
│  ┌──────┴─────┐   ┌──────┴───────┐   ┌──────┴───────┐  │
│  │SKILL: ETL  │   │SKILL: RAG    │   │SKILL:Analytic│  │
│  │ SKILL.md   │   │ SKILL.md     │   │ SKILL.md     │  │
│  │ ────────── │   │ ──────────── │   │ ──────────── │  │
│  │ •parse()   │   │ •search()    │   │ •query()     │  │
│  │ •chunk()   │   │ •rerank()    │   │ •visualize() │  │
│  │ •embed()   │   │ •generate()  │   │ •report()    │  │
│  └────────────┘   └──────────────┘   └──────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │          Privacy Router + NeMo Guardrails         │    │
│  │  local Nemotron ◄──► policy gate ──► cloud LLM    │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │          Claw Scheduler (always-on / cron)        │    │
│  │  • nightly batch ingest    • anomaly alerts       │    │
│  │  • daily report generation • model health checks  │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

### 5.2 Skill Definitions (Example)

Each expert maps to an OpenClaw **Skill** directory:

```
skills/
├── ocr-agent/
│   └── SKILL.md          # Instructions for OCR tool usage
├── object-detection/
│   └── SKILL.md          # Cargo detection prompts & tool calls
├── multimodal-reasoning/
│   └── SKILL.md          # VLM inference routing
├── data-pipeline/
│   └── SKILL.md          # PDF→CSV/JSON→Embed pipeline
├── rag-retrieval/
│   └── SKILL.md          # Vector search + reranking
└── analytics/
    └── SKILL.md          # SQL generation + dashboarding
```

### 5.3 NemoClaw Deployment Flow

```
1. Install NemoClaw (single command)
   └─► Pulls Nemotron-3 Nano + OpenShell runtime

2. Register Skills
   └─► Each expert's SKILL.md is loaded into the Skill Registry

3. Configure Privacy Router
   └─► Define which data stays local vs. can hit cloud APIs
   └─► Logistics PII (shipment IDs, customer data) → local only

4. Set Guardrails
   └─► NeMo Guardrails filter PII leakage, prompt injection
   └─► Output validation per expert (e.g., OCR confidence thresholds)

5. Schedule Always-On Claws
   └─► Nightly: batch ingest new shipping docs
   └─► Hourly: monitor for anomaly alerts
   └─► On-demand: user queries via chat interface
```

---

## 6. API Server Trade-offs

### 6.1 Framework Choice

| Factor | FastAPI (Python) | Express/Hono (Node) |
|--------|-----------------|---------------------|
| **ML ecosystem** | Native (PyTorch, HF, vLLM) | Requires Python sidecar |
| **Async I/O** | Good (asyncio) | Excellent (event loop) |
| **Type safety** | Pydantic v2 | TypeScript + Zod |
| **Next.js integration** | Separate service | Same runtime possible |
| **Team skill alignment** | ML engineers | Full-stack JS devs |

**Recommendation:** **FastAPI** for the inference/agent layer (direct access to Python ML stack); **Next.js API routes** for auth and client-facing BFF.

### 6.2 Key Trade-offs for Locally-Hosted Inference

| Trade-off | Consideration |
|-----------|--------------|
| **VRAM budget** | Nemotron-3 Nano FP8 ≈ 30GB; Qwen2.5-VL-7B ≈ 16GB; need 48–80GB total for all experts concurrently |
| **Latency vs. throughput** | vLLM continuous batching helps, but MoE routing adds ~10-50ms overhead per request |
| **Model hot-swapping** | Keep high-use experts warm; cold-load rarely-used ones (analytics) on demand |
| **Data privacy** | All customer data stays local; NemoClaw Privacy Router enforces this |
| **Scaling** | Vertical first (single multi-GPU node); horizontal later via Triton + load balancer |
| **Fine-tuning cost** | 30M tokens ≈ 1-3 epochs of LoRA fine-tuning on Nemotron Nano; ~4-8 GPU-hours on A100 |
| **Embedding refresh** | Re-embed on schema changes; incremental upsert for new documents |
| **Guardrails overhead** | NeMo Guardrails add ~20-50ms per call; acceptable for non-real-time queries |

### 6.3 Hardware Recommendations

| Tier | Hardware | Capacity |
|------|----------|----------|
| **Dev/POC** | 2× RTX 3090 (48GB) | Nemotron Nano FP8 + 1 vision model | [https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
| **Production (Minimum)** | 1× A100 80GB or 2× RTX 6000 Ada | All experts concurrently |
| **Production (Recommended)** | DGX Station / 4× A100 | Full stack + fine-tuning headroom |
| **Enterprise** | DGX Spark / DGX H100 | Multi-tenant, always-on claws |

---

## 7. Data Pipeline Proposal

```
                    ┌─────────────────────────────────────┐
                    │       RAW CUSTOMER DATA (~30M tok)   │
                    │   PDFs │ Images │ CSVs │ JSON │ XML  │
                    └─────────┬───────────────────────────┘
                              │
                    ┌─────────▼───────────────────────────┐
                    │     1. DOCUMENT PARSING               │
                    │  Unstructured.io / Docling / LlamaParse│
                    │  → structured markdown / JSON         │
                    └─────────┬───────────────────────────┘
                              │
                    ┌─────────▼───────────────────────────┐
                    │     2. OCR (if scanned)               │
                    │  GOT-OCR2 / PaddleOCR                │
                    │  → extracted text with confidence     │
                    └─────────┬───────────────────────────┘
                              │
                    ┌─────────▼───────────────────────────┐
                    │     3. OBJECT DETECTION (if images)   │
                    │  Grounding DINO / OWLv2              │
                    │  → bounding boxes + labels + metadata │
                    └─────────┬───────────────────────────┘
                              │
                    ┌─────────▼───────────────────────────┐
                    │     4. CHUNKING                       │
                    │  Semantic chunking (512-1024 tokens)  │
                    │  Preserve table/list structure        │
                    │  Attach metadata (source, page, date) │
                    └─────────┬───────────────────────────┘
                              │
                    ┌─────────▼───────────────────────────┐
                    │     5. EMBEDDING                      │
                    │  NV-Embed-v2 (4096-dim)              │
                    │  Batch processing via vLLM            │
                    └─────────┬───────────────────────────┘
                              │
                    ┌─────────▼───────────────────────────┐
                    │     6. QUALITY VALIDATION             │
                    │  Deduplication (MinHash / SimHash)    │
                    │  Schema validation                    │
                    │  Confidence thresholding              │
                    └─────────┬───────────────────────────┘
                              │
              ┌───────────────┼───────────────────┐
              ▼               ▼                   ▼
      ┌──────────────┐ ┌────────────┐  ┌──────────────────┐
      │ Vector DB    │ │ Doc Store  │  │ Metadata DB      │
      │ (Milvus/     │ │ (MongoDB/  │  │ (PostgreSQL)     │
      │  Qdrant)     │ │  S3)       │  │                  │
      │ embeddings + │ │ raw docs + │  │ lineage, audit,  │
      │ metadata     │ │ parsed out │  │ schema, stats    │
      └──────────────┘ └────────────┘  └──────────────────┘
```

---

## 8. Authentication Flow

```
Client (Next.js)
    │
    ├──► /api/auth/login  ──► NextAuth.js / Keycloak
    │         │
    │         ├──► Validate credentials
    │         ├──► Issue JWT (access + refresh tokens)
    │         ├──► Store session in Redis
    │         └──► Return token + RBAC scopes
    │
    ├──► /api/query  ──► API Gateway (FastAPI)
    │         │
    │         ├──► Verify JWT signature
    │         ├──► Check RBAC scope for requested expert
    │         ├──► Route to MoE Router
    │         └──► Return response
    │
    └──► /api/upload ──► API Gateway
              │
              ├──► Verify JWT + upload scope
              ├──► Stream to ingestion pipeline
              └──► Return processing status
```

---

## 9. Estimated Resource Requirements

| Component | Memory | Storage | Notes |
|-----------|--------|---------|-------|
| Nemotron-3 Nano (FP8) | ~30GB VRAM | ~30GB disk | Central reasoning model |
| Qwen2.5-VL-7B (FP16) | ~16GB VRAM | ~15GB disk | Multi-modal expert |
| Grounding DINO | ~2GB VRAM | ~1.5GB disk | Object detection |
| GOT-OCR2.0 | ~4GB VRAM | ~2GB disk | OCR |
| NV-Embed-v2 | ~2GB VRAM | ~1.5GB disk | Embeddings |
| Reranker | ~1GB VRAM | ~0.5GB disk | Reranking |
| Vector DB (30M tokens) | ~8GB RAM | ~20GB disk | ~60k chunks × 4096-dim |
| **Total (concurrent)** | **~63GB VRAM** | **~70GB disk** | Fits on DGX Station / 2×A100 |

---

## 10. Next Steps

1. **Validate hardware** — confirm client GPU/server inventory
2. **Data audit** — sample 30M token dataset for format distribution (% PDF, % image, % structured)
3. **POC** — stand up Nemotron-3 Nano + 1 expert (RAG) end-to-end
4. **Benchmark** — compare OCR/detection model accuracy on client's actual shipping docs
5. **NemoClaw setup** — install NemoClaw, define skills, configure privacy router
6. **Auth scaffold** — Next.js + NextAuth.js with JWT flow
7. **Full MoE rollout** — add remaining experts iteratively
8. **Fine-tuning** — LoRA on Nemotron Nano with client's domain data
9. **Load testing** — concurrent query benchmarks
10. **Production hardening** — guardrails, monitoring, CI/CD

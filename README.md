# StudyMind AI — Personal AI Learning Workspace

## 📌 Overview
StudyMind AI is a personal AI-powered learning workspace where students can upload learning material from multiple sources — PDFs, YouTube lectures, web articles, and personal notes — and let the system understand it, answer questions, generate study aids, and guide their learning journey.

This is **not** a simple chatbot. It's an intelligent learning ecosystem that:
- Understands content from multiple sources (PDF, YouTube, articles, notes)
- Builds a knowledge graph connecting related concepts
- Answers questions using Retrieval-Augmented Generation (RAG)
- Auto-generates flashcards and quizzes
- Identifies weak topics and builds a personalized study roadmap
- Remembers past conversations and context over time

## 🎯 Objectives
1. Let users bring learning material from multiple sources into one place
2. Process and understand that material using AI (not just store it)
3. Enable users to "talk" to their own material via a RAG-based chatbot
4. Automatically generate study aids (flashcards, quizzes)
5. Track weak topics and recommend what to study next
6. Maintain long-term memory across sessions

## 🧩 Project Structure

```
studymind-ai/
├── web/          → Web Development Team (Team Pluto)
│   ├── frontend/ → React app (upload UI, chat UI, dashboard)
│   └── backend/  → FastAPI backend, auth, main API
│
├── ai-ml/        → AI/ML Team (Team Lambda)
│   ├── ingestion/       → PDF/YouTube/article text extraction
│   ├── embeddings/      → Vector database logic
│   └── quiz-generator/  → Flashcard & quiz generation
│
├── chatbot/      → Chatbot Team (Team Mu)
│   ├── rag-engine/  → Retrieval-Augmented Generation pipeline
│   └── memory/       → Conversation memory across sessions
│
└── docs/         → Shared documentation for all teams
    ├── architecture.md    → How all modules connect
    ├── api-contracts.md   → Data formats passed between modules
    └── meeting-notes.md   → Team meeting notes
```

## 👥 Teams & Responsibilities

| Team | Folder | Responsibility |
|------|--------|-----------------|
| **Team Pluto** | `web/` | Frontend UI, backend API, authentication, dashboard |
| **Team Lambda** | `ai-ml/` | Content ingestion, embeddings, vector search, quiz generation |
| **Team Mu** | `chatbot/` | RAG-based Q&A engine, conversation memory |

**Important rule:** Each team works **only** inside their assigned folder. Cross-team communication about data formats happens through `docs/api-contracts.md`, not by editing each other's code directly.

## 🌱 Development Approach
This project is being built step by step, starting with small, simple tasks and gradually increasing in complexity as the team grows. Early tasks focus on getting the basic plumbing working (e.g., a working frontend-backend connection, basic text extraction, a simple LLM call) before adding intelligence and advanced features.

## 🔀 Git Workflow
- `main` branch is always stable — no one pushes directly to it
- Every task is done in its own branch, named like `team/task-name` (e.g. `web/upload-ui`, `ai-ml/pdf-parser`)
- Work is submitted via Pull Request and reviewed before merging
- Branch protection is enabled on `main` to enforce this

## 🛠️ Tech Stack (proposed)
- **Frontend:** React + Vite + Tailwind
- **Backend:** FastAPI (Python)
- **Database:** MongoDB / PostgreSQL
- **Vector Database:** ChromaDB / Qdrant
- **LLM:** Anthropic / OpenAI API
- **OCR:** Tesseract / PyMuPDF
- **YouTube Transcripts:** youtube-transcript-api / Whisper

## 🚀 Getting Started
Setup instructions will be added here as each module becomes functional.



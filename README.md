# AI Document Q&A (RAG prototype)

Ask questions about a PDF and get answers drawn from the document itself. The system finds the most relevant
passages, hands them to a locally run LLM, and tells it to answer **only from those passages**, or to say
"I don't know".

This is a **first prototype**, built as a team Design Thinking project. It runs end to end and taught us where
retrieval breaks; the results and next steps below are written up honestly.

**Team RAG:** Abel James Hardass · Jacob James · G. Sai Mahin

---

## Why we built it

Generative AI gives fluent answers that can be wrong ("hallucinations"), which is a real risk wherever people act
on the answer. We ran **4 interview sessions (40 questions)** with a client who researches and teaches blockchain.
What they told us shaped the design:

- **Hallucinations were the top concern.** "Inaccurate AI outputs could mislead researchers."
- **Answers must point back to sources.** Summaries should come with citations to the original papers.
- **Filtering long documents is the slow part.** Reading whitepapers and pulling out key findings takes most of
  their time.

So the goal: answers grounded in a specific document, with a clear "I don't know" when the document does not say.

The full process (stakeholders, personas, POV and "How might we" statements, ideation, solution concept) is in
[`DTI documentation.pdf`](AIML_RAG_PROJECT-main/DTI%20documentation.pdf); the interview notes are in
[`cilent meetings.pdf`](AIML_RAG_PROJECT-main/cilent%20meetings.pdf); the one-page brief is in
[`Ai_ml project_abstract.pdf`](AIML_RAG_PROJECT-main/Ai_ml%20project_abstract.pdf).

## How it works

```
PDF ──► split into pages ──► embed each page ──► in-memory vector store
                                                      │
question ──► embed ──► find the 4 most similar pages ─┘
                                │
                                ▼
   prompt: "Answer based on the following text. If you don't know, say 'I don't know'."
                                │
                                ▼
                    Llama 2 (local, via Ollama) ──► answer
```

| Step | What the notebook uses |
|---|---|
| Load the document | LangChain `PyPDFLoader` (`load_and_split`, roughly one chunk per page) |
| Embeddings | `OllamaEmbeddings` (Llama 2) |
| Vector store | `DocArrayInMemorySearch` (in memory, rebuilt each run) |
| Retrieval | similarity search, top 4 chunks |
| Answer | Llama 2 through Ollama, with a grounding prompt and an "I don't know" fallback |
| Test document | *Alice's Adventures in Wonderland* (66-page PDF) |

Everything runs **locally**: no API key, no data leaves the machine.

## Repository contents

```
AIML_RAG_PROJECT-main/
├── AIML_project/
│   ├── notebook.ipynb            # the whole pipeline + the 20-question test
│   └── alice-in-wonderland.pdf   # test document
├── Ai_ml project_abstract.pdf    # project abstract
├── DTI documentation.pdf         # Design Thinking workbook
└── cilent meetings.pdf           # client interview notes
```

## Run it

1. Install [Ollama](https://ollama.com) and pull the model: `ollama pull llama2`
2. Create a Python environment and install the libraries:
   ```bash
   pip install langchain langchain-community pypdf docarray python-dotenv jupyter
   ```
3. Open `AIML_RAG_PROJECT-main/AIML_project/notebook.ipynb` and run the cells top to bottom (Ollama must be running).

## First test: 20 questions

We asked 20 questions about the book, from simple facts to themes, plus trick questions it should refuse.
Each answer was then read against the book:

| Outcome | Questions |
|---|---|
| Correctly declined (the book doesn't say, or a trick question) | 3 |
| Partly right | 3 |
| Missed: said "I don't know" although the answer is in the book | 6 |
| Wrong, or not grounded in the text | 8 |

Examples:
- *"Who was the author of Alice in Wonderland and The Hobbit?"* → "I don't know." Good: the guardrail refuses a
  trick question instead of inventing an answer.
- *"What role does the Dormouse play in the Mad Hatter's tea party?"* → "I don't know." A miss: the tea party is
  in the book, but retrieval did not bring that page.
- *"Who gives Alice the advice 'We're all mad here'?"* → "the Caterpillar". Wrong: it is the Cheshire Cat.

**What this told us:** the "I don't know" fallback works, but **retrieval is the weak link**. Whole pages are coarse
chunks, a chat model is a weak embedding model, and the model sometimes answers from its own memory instead of
the retrieved text.

## Next steps

- [ ] Smaller overlapping chunks (`RecursiveCharacterTextSplitter`), as planned in the abstract
- [ ] A dedicated embedding model (for example `nomic-embed-text`) instead of Llama 2 embeddings
- [ ] Persistent vector store (ChromaDB, as planned) so documents are not re-embedded on every run
- [ ] Cite the page number behind every answer (the client's main request)
- [ ] An answer key for the 20 questions, so each change is measured instead of eyeballed
- [ ] A newer local model, plus a small UI to upload a PDF and ask questions

## Original plan vs. this prototype

The abstract planned OpenAI's API with ChromaDB and recursive text splitting. The prototype runs fully local
instead (Llama 2 through Ollama, in-memory search, page-level chunks), which makes the planned splitting and
ChromaDB the obvious next steps.

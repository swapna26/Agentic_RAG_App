# Agentic RAG Evaluator (CrewAI + RAGas)

A comprehensive Agentic RAG evaluation system that uses CrewAI agents to generate answers and the RAGas framework to score them.

## Features

- **Automatic Test Dataset Generation**: Generate test questions from indexed documents
- **Comprehensive RAG Evaluation**: Evaluate using multiple RAGAs metrics
- **Custom Question Support**: Evaluate with your own custom questions
- **Detailed Reporting**: Generate comprehensive evaluation reports with recommendations
- **CLI Interface**: Easy-to-use command-line interface
- **Rich Console Output**: Beautiful formatted output with progress tracking

## RAGAs Metrics Supported

1. **Faithfulness**: Measures factual consistency of generated answers
2. **Answer Relevancy**: Evaluates relevance of answers to questions
3. **Context Precision**: Measures precision of retrieved context
4. **Context Recall**: Evaluates recall of relevant context
5. **Answer Similarity**: Compares semantic similarity of answers
6. **Answer Correctness**: Evaluates factual and semantic correctness

## Installation

1. Install dependencies:
```bash
cd evaluator
pip install -r requirements.txt
```

## Configuration

The evaluator automatically connects to your existing infrastructure:
- **PostgreSQL Database**: Same database used by the indexer
- **Ollama Models**: Uses the same embedding and LLM models
- **Backend API**: Connects to your RAG backend API

Configuration is handled in `config.py` with these defaults:
- Database: `postgresql://raguser:ragpassword@localhost:5432/agentic_rag`
- Ollama: `http://localhost:11434`
- Backend API: `http://localhost:8001`
- Embedding Model: `nomic-embed-text:v1.5`
- LLM Model: `llama3.2:1b`

## Usage (Step-by-step flow)

### 1) Check system status
```bash
python main.py status
```

### 2) Generate test dataset (optional)
```bash
# Generate 20 test questions from indexed documents
python main.py generate-dataset

# Generate custom number of questions
python main.py generate-dataset --num-questions 50

# Save to custom location
python main.py generate-dataset --output /path/to/dataset.json
```

### 3) Run evaluation (CrewAI answers + RAGas metrics)
```bash
# Run complete evaluation with all metrics
python main.py evaluate

# Evaluate specific metrics only
python main.py evaluate --metrics faithfulness --metrics answer_relevancy

# Evaluate with custom questions
python main.py evaluate --questions "What is the procurement process?" --questions "How to handle HR issues?"

# Use custom dataset
python main.py evaluate --dataset /path/to/custom_dataset.json

# Save report to custom location
python main.py evaluate --output /path/to/report.json
```

### 4) View reports
```bash
# List all evaluation reports
python main.py list-reports

# View specific report
python main.py show-report --report /path/to/report.json
```

## CLI Commands

### `generate-dataset`
Generate test questions from indexed documents.

**Options:**
- `--num-questions, -n`: Number of questions to generate (default: 20)
- `--output, -o`: Output path for the dataset file

### `evaluate`
Run RAGAs evaluation on the RAG system.

**Options:**
- `--dataset, -d`: Path to test dataset JSON file
- `--metrics, -m`: Specific metrics to evaluate (can be repeated)
- `--output, -o`: Output path for evaluation report
- `--questions, -q`: Custom questions to evaluate (can be repeated)

### `status`
Check the status of the evaluation system.

### `show-report`
Display evaluation report in formatted output.

**Options:**
- `--report, -r`: Path to evaluation report JSON file (required)

### `list-reports`
List available evaluation reports.

**Options:**
- `--path, -p`: Path to reports directory

## Example end-to-end flow (call-by-call)

1) **Check if your system is ready:**
```bash
python main.py status
```

2) **Generate test dataset:**
```bash
python main.py generate-dataset --num-questions 30
```

3) **Run evaluation:**
```bash
python main.py evaluate
```

4) **View the results:**
```bash
python main.py list-reports
python main.py show-report --report reports/rag_evaluation_report_20240914_120000.json
```

## Output Structure

### Test Dataset
Generated test datasets contain:
- Questions with different types (factual, conceptual, analytical, procedural)
- Ground truth answers
- Source document references
- Context information

### Evaluation Reports
Comprehensive JSON reports include:
- Metric scores with statistics (mean, std, min, max)
- Individual question results
- System configuration and metadata
- Performance summary and recommendations
- Database statistics

## Advanced Usage

### Custom Metrics
Modify `config.py` to select specific metrics:
```python
config.metrics = ["faithfulness", "answer_relevancy", "context_precision"]
```

### Custom Configuration
Override default settings in `config.py`:
```python
config.backend_api_url = "http://your-backend:8000"
config.database_url = "postgresql://user:pass@host:port/db"
config.num_test_questions = 50
```

## Current issues (today) and mitigations

1) Some metrics return NaN or time out (faithfulness, answer_relevancy, context_precision, answer_correctness)
- Symptom: Progress bar shows many TimeoutError events; final report has 0/NaN for these metrics.
- Cause: These metrics execute multi-step LLM/NLI prompts. With RAGas 0.3.4 + LlamaIndex wrappers over Ollama, they frequently exceed time limits.
- Mitigations we applied (already in code):
  - Deterministic LLM config for evaluator: temperature=0.0, num_predict=512, request_timeout=60s
  - Guarded evaluation with a global timeout and robust error logging
  - Confirmed dataset schema: question, answer, contexts (list[str]), ground_truth
- Options to get all metrics today (choose one):
  - Keep CrewAI for answers but switch evaluator’s scoring LLM to a faster API (e.g., OpenAI) only for metrics; or
  - Upgrade to RAGas >= 0.4 which improves metric execution and compatibility.


### Performance tips

- Start with a smaller number of test questions (10-20) for faster evaluation
- Use specific metrics instead of all metrics for quicker results
- Generate test datasets once and reuse them for multiple evaluations

## Integration

The evaluator is designed to work seamlessly with your existing Agentic RAG setup:
- Reads from the same PostgreSQL vector database
- Uses the same Ollama models and configuration
- Connects to your backend API for RAG queries
- Generates reports in the `reports/` directory

No changes to other components are required!

## Issues & Resolutions (Field Notes)

This section documents the exact issues we ran into during setup and how we resolved each one, in order. Use it as a checklist when reproducing or presenting.

1) Timeouts during heavy metrics (faithfulness, relevancy, precision, correctness)
- Cause: These metrics perform multi-step LLM/NLI operations; Ollama via wrappers can be slow.
- Fixes:
  - Lowered generation (`num_predict=512`) and `temperature=0.0`.
  - Added evaluation timeout guards.
  - For stable demos: run with a subset of compatible metrics (answer_similarity, context_recall). For all metrics, consider faster model backends or RAGas >= 0.4.

## Playbooks

### A) Stable demo (CrewAI answers + RAGas, compatible metrics)
```bash
cd evaluator
python3 main.py evaluate -d datasets/test_questions.json \
  -m answer_similarity -m context_recall \
  -o reports/crew_ai_eval_stable.json
python3 main.py show-report -r reports/crew_ai_eval_stable.json
```

Expected: Non‑NaN numbers; recent run achieved answer_similarity ≈ 0.825 and context_recall ≈ 0.722.

### B) Full metrics run (may be slow or partially NaN on Ollama)
```bash
cd evaluator
python3 main.py evaluate -d datasets/test_questions.json \
  -m faithfulness -m answer_relevancy -m context_precision -m context_recall -m answer_similarity -m answer_correctness \
  -o reports/crew_ai_eval_full_metrics.json
python3 main.py show-report -r reports/crew_ai_eval_full_metrics.json
```

Tip: If LLM metrics time out, reduce question count, limit contexts to top 2, or switch evaluator’s scoring LLM to a faster provider while keeping CrewAI as the answer source.
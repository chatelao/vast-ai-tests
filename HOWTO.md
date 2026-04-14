# HOWTO: Benchmarking LLMs on Vast.ai

This guide describes how to benchmark LLMs on Vast.ai using this automated framework. The recommended approach is to use the provided GitHub Action workflow, which handles the entire lifecycle from provisioning to cleanup.

## Primary Workflow: GitHub Actions (Automated)

Running benchmarks via GitHub Actions is the most reliable and efficient method. The orchestrator runs on the GitHub runner and manages a remote GPU instance on Vast.ai.

### 1. Prerequisites
- **Vast.ai API Key:** Get your API key from the [Vast.ai Console](https://vast.ai/console/billing/).
- **HuggingFace Token:** For gated models (like Gemma), get a token from [HuggingFace Settings](https://huggingface.co/settings/tokens).
- **GitHub Secrets:** Add `VAST_AI_API_KEY` and `HF_TOKEN` to your repository's GitHub Actions secrets.

### 2. Running the Benchmark
- Go to the **Actions** tab in your GitHub repository.
- Select the **Vast.ai LLM Benchmark** workflow.
- Click **Run workflow**.
- Configure the inputs:
    - **GPU model:** e.g., `RTX_4090`, `A100`.
    - **LLM model:** Choose from the predefined list.
    - **Concurrency levels:** Space-separated list (e.g., `1 4 16`).
    - **Template Hash:** Use the default optimized vLLM template.
- Click **Run workflow**.

### 3. Viewing Results
- Once the workflow completes, the results table will be displayed in the **GitHub Step Summary**.
- Detailed JSON results are available as a workflow artifact named `benchmark-results`.
- The instance is automatically destroyed at the end of the workflow, even if a failure occurs.

## Manual Workflow (CLI-based)

You can also run the orchestrator manually from your local machine.

### 1. Installation
```bash
pip install -r requirements.txt
export VAST_AI_API_KEY=your_key
export HF_TOKEN=your_hf_token
```

### 2. Execution
The orchestrator will provision an instance using the optimized vLLM template, run the benchmarks, and then destroy the instance.

```bash
python3 provision.py --gpu "RTX_4090" --model "google/gemma-2-9b-it"
python3 benchmark.py --gpu "RTX_4090" --model "google/gemma-2-9b-it"
python3 teardown.py
```

The framework provides granular execution scripts, which are used by the GitHub Actions workflow:
- `provision.py`: Rent the instance and wait for the API to be ready.
- `benchmark.py`: Run the benchmark suite against an already provisioned instance.
- `teardown.py`: Destroy the provisioned instance.

## Local Verification (No GPU required)

To verify the scripting logic without renting a GPU, use the micro runtime test.

```bash
pip install vllm-cpu --extra-index-url https://download.pytorch.org/whl/cpu
python tests/micro_runtime_test.py
```

## Architecture Details

This framework relies on optimized Vast.ai templates. The default template (`38b2b68cf896e8582dff6f305a2041b1`) is pre-configured with:
- **vLLM Engine:** For high-throughput inference.
- **Auto-Config:** Uses environment variables (`VLLM_MODEL`, `HF_TOKEN`, `OPEN_BUTTON_TOKEN`) for zero-touch setup.
- **Port Mapping:** Exposes services on standard ports:
    - Instance Portal: 1111
    - Model UI: 7860
    - vLLM API: 8000
    - Ray Dashboard: 8265
    - Jupyter: 8080

# HOWTO: From Zero to Test Results (On-Instance)

This guide provides a step-by-step checklist to benchmark LLMs directly on a Vast.ai instance. Running benchmarks on-instance eliminates network latency and provides the most accurate hardware metrics.

## Phase 1: Preparation
- [ ] **Vast.ai Account:** Create an account at [vast.ai](https://vast.ai/).
- [ ] **Add Credits:** Ensure your account has a balance to rent instances.
- [ ] **HuggingFace Token:** If testing gated models (like Gemma), get a token from [huggingface.co](https://huggingface.co/settings/tokens).

## Phase 2: Renting an Instance
- [ ] **Search for Hardware:** Go to the [Vast.ai Console](https://vast.ai/console/create/) and search for your desired GPU (e.g., "RTX 4090").
- [ ] **Rent:** Select an offer and rent the instance.
- [ ] **Get SSH Details:** Once the instance is ready, note the SSH IP and Port from the "Instances" tab.

## Phase 3: On-Instance Setup & Benchmarking
- [ ] **SSH into the instance:**
  ```bash
  ssh -p <PORT> root@<IP>
  ```
- [ ] **Clone the Repository:**
  ```bash
  git clone <repo-url>
  cd avast-ai-tests
  ```
- [ ] **Install Dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
- [ ] **Start LLM Engine:** (In a separate terminal or background)
  ```bash
  docker run --gpus all \
             -v ~/.cache/huggingface:/root/.cache/huggingface \
             -e HF_TOKEN=<your_token> \
             -p 8000:8000 vllm/vllm-openai:v0.4.0 \
             --model google/gemma-2-9b-it
  ```
- [ ] **Run Orchestrator:**
  ```bash
  python3 orchestrator.py --gpu "RTX 4090" --model "gemma-2-9b-it" --url http://localhost:8000 --run
  ```

## Phase 4: Analyze Results
- [ ] **Check Output File:** Results are saved as `benchmark_<gpu>_<timestamp>.json`.
- [ ] **Verify Metrics:**
    - **TTFT:** Time to First Token (lower is better).
    - **ITL:** Inter-Token Latency (lower is better).
    - **TPS:** Tokens Per Second (higher is better).

## Phase 5: Cleanup
- [ ] **Destroy Instance:** Go back to the [Vast.ai Console](https://vast.ai/console/instances/) and destroy the instance to prevent unwanted charges.

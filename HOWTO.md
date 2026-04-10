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

## Phase 3: Optimized On-Instance Setup & Benchmarking
To minimize runtime on expensive hardware, we maximize startup efficiency by running the LLM engine and setup tasks in parallel.

- [ ] **SSH and Start LLM Engine Immediately:**
  Run the engine in detached mode (`-d`) so it starts pulling the model while you finish setup.
  ```bash
  ssh -p <PORT> root@<IP>
  # Recommendation: use the module entrypoint for better stability in some environments
  # python -m vllm.entrypoints.openai.api_server --model google/gemma-2-9b-it ...
  docker run -d --gpus all \
             -v ~/.cache/huggingface:/root/.cache/huggingface \
             -e HF_TOKEN=<your_token> \
             -p 8000:8000 vllm/vllm-openai:v0.4.0 \
             --model google/gemma-2-9b-it \
             --max-model-len 512 \
             --block-size 16 \
             --dtype float \
             --enforce-eager
  ```
- [ ] **Setup Benchmarking Tools (Parallel to Download):**
  While the Docker image is pulling in the background, clone the repo and install requirements.
  ```bash
  git clone <repo-url>
  cd avast-ai-tests
  pip install -r requirements.txt
  ```
- [ ] **Run Orchestrator with Auto-Ready:**
  The orchestrator will automatically wait for the API to be ready before starting the benchmark.
  ```bash
  python3 orchestrator.py --gpu "RTX 4090" --model "gemma-2-9b-it" --url http://localhost:8000 --run
  ```

## Phase 4: Analyze Results & Immediate Cleanup
To avoid unnecessary charges, analyze your results and destroy the instance immediately.
- [ ] **Check Output File:** Results are saved as `benchmark_<gpu>_<timestamp>.json`.
- [ ] **Verify Metrics:**
    - **TTFT:** Time to First Token (lower is better).
    - **ITL:** Inter-Token Latency (lower is better).
    - **TPS:** Tokens Per Second (higher is better).

## Phase 5: Fast Cleanup
- [ ] **Destroy Instance:** Immediately go back to the [Vast.ai Console](https://vast.ai/console/instances/) and destroy the instance. For expensive hardware, every minute counts.

## Pro Tip: Using Templates for Even Faster Setup
Using a **Vast.ai Template** (configured via the Web Console) is significantly faster than manual `docker run` because:
- **Instant Pull:** The host starts pulling the vLLM image the moment the instance is provisioned.
- **Zero-Touch:** You can configure the `docker run` command and environment variables (like `HF_TOKEN`) directly in the template.
- **Parallelism:** The engine starts while you are still SSHing in to clone the benchmarking repo.

To use a template with our tools, find your template's **Hash ID** in the Vast.ai console and pass it to the orchestrator (if supported) or use it in `vast_manager.py`:
```python
mgr.rent_instance(offer_id, template_hash="your_template_hash")
```

# HOWTO: From Zero to Test Results

This guide provides a step-by-step checklist to get you from a fresh environment to benchmarking LLMs on Vast.ai.

## Phase 1: Preparation
- [ ] **Vast.ai Account:** Create an account at [vast.ai](https://vast.ai/).
- [ ] **Add Credits:** Ensure your account has a balance to rent instances.
- [ ] **API Key:** Retrieve your API key from the [Vast.ai Console](https://vast.ai/console/billing/).
- [ ] **HuggingFace Token:** If testing gated models (like Gemma), get a token from [huggingface.co](https://huggingface.co/settings/tokens).

## Phase 2: Local Environment Setup
- [ ] **Clone the Repository:**
  ```bash
  git clone <repo-url>
  cd avast-ai-tests
  ```
- [ ] **Install Dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
- [ ] **Configure Vast.ai CLI:**
  ```bash
  vastai set api-key <YOUR_VAST_AI_API_KEY>
  ```

## Phase 3: GPU Discovery
- [ ] **Search for Hardware:** Find available GPUs and their Offer IDs.
  ```bash
  python3 infra/vast_manager.py --search "RTX 4090"
  ```
- [ ] **Note the Offer ID:** Identify the ID of the instance you want to rent from the output.

## Phase 4: Deployment & Benchmarking
You can use the Orchestrator for an automated flow, but note that it currently requires manual steps for the LLM engine deployment.

### Option A: Manual Flow (Recommended for full control)
- [ ] **Rent an Instance:**
  ```bash
  python3 infra/vast_manager.py --rent <OFFER_ID>
  ```
- [ ] **Wait for SSH:** The script will output the SSH IP and Port once ready.
- [ ] **Deploy LLM Engine:** SSH into the machine or use the `docker run` command provided in `README.md` to start vLLM.
  ```bash
  docker run --gpus all \
             -v ~/.cache/huggingface:/root/.cache/huggingface \
             -e HF_TOKEN=<your_token> \
             -p 8000:8000 vllm/vllm-openai:v0.4.0 \
             --model google/gemma-2-9b-it
  ```
- [ ] **Run Load Test:**
  ```bash
  python3 bench/load_tester.py --url http://<instance-ip>:8000 --model gemma-2-9b-it --concurrency 10 --requests 50
  ```

### Option B: Automated Orchestrator
- [ ] **Run Orchestrator:**
  ```bash
  python3 orchestrator.py --gpu "RTX_4090" --model "gemma-2-9b" --run
  ```
  *Note: You will need to manually start the LLM server on the instance when the script prompts you.*

### Option C: Local Benchmarking (Run on the Vast.ai instance)
For the most accurate hardware metrics, you can run the benchmarking tools directly on the rented instance. This eliminates network latency between your local machine and the instance.

- [ ] **SSH into the instance:** Use the credentials provided when renting.
- [ ] **Setup Environment:**
  ```bash
  git clone <repo-url>
  cd avast-ai-tests
  pip install -r requirements.txt
  ```
- [ ] **Start LLM Engine:** (In a separate terminal or background)
  ```bash
  docker run --gpus all -p 8000:8000 vllm/vllm-openai --model google/gemma-2-9b-it
  ```
- [ ] **Run Orchestrator with Local URL:**
  ```bash
  python3 orchestrator.py --gpu "RTX 4090" --model "gemma-2-9b-it" --url http://localhost:8000 --run
  ```

## Phase 5: Analyze Results
- [ ] **Check Output File:** Results are saved as `benchmark_<gpu>_<timestamp>.json`.
- [ ] **Verify Metrics:**
    - **TTFT:** Time to First Token (lower is better).
    - **ITL:** Inter-Token Latency (lower is better).
    - **TPS:** Tokens Per Second (higher is better).

## Phase 6: Cleanup
- [ ] **Destroy Instance:** Prevent unwanted charges.
  ```bash
  python3 infra/vast_manager.py --destroy <INSTANCE_ID>
  ```

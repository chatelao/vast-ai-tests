# HOWTO: From Zero to LLM Test Results on Vast.ai

This guide provides a step-by-step checklist to set up and run LLM performance benchmarks using the `avast-ai-tests` framework.

## 1. Preparation

- [ ] **Vast.ai Account:** Create an account at [vast.ai](https://vast.ai/).
- [ ] **Add Credits:** Ensure your account has sufficient credits to rent GPU instances.
- [ ] **API Key:** Obtain your API Key from the [Vast.ai Console](https://vast.ai/console/billing/).
- [ ] **SSH Key:** Add your public SSH key to your Vast.ai account settings. This is required to access the instances you rent.

## 2. Local Environment Setup

- [ ] **Clone Repository:**
  ```bash
  git clone <repository-url>
  cd avast-ai-tests
  ```
- [ ] **Install Python:** Ensure Python 3.10+ is installed.
- [ ] **Install Dependencies:**
  ```bash
  pip install -r requirements.txt
  ```

## 3. Configuration

- [ ] **Configure Vast.ai CLI:**
  ```bash
  vastai set api-key <YOUR_API_KEY>
  ```
- [ ] **Verify Connectivity:**
  ```bash
  python3 infra/vast_manager.py --search "RTX 4090"
  ```
  This should return a list of available RTX 4090 offers.

## 4. Execution

### Option A: Automated Orchestration (Recommended)

The orchestrator automates finding, renting, and cleaning up instances.

1. [ ] **Run the Orchestrator (Dry Run):**
   ```bash
   python3 orchestrator.py --gpu "RTX_4090" --model "gemma-2-9b"
   ```
2. [ ] **Run the Full Pipeline:**
   ```bash
   python3 orchestrator.py --gpu "RTX_4090" --model "gemma-2-9b" --run
   ```
   *Note: The script will rent a real instance and charge your account.*
3. [ ] **Manual Step - Deploy LLM Engine:**
   When the orchestrator provides the SSH host and port, you must manually SSH into the instance and start the LLM engine (e.g., vLLM).
   ```bash
   docker run --gpus all \
              -v ~/.cache/huggingface:/root/.cache/huggingface \
              -e HF_TOKEN=<your_token> \
              -p 8000:8000 vllm/vllm-openai:v0.4.0 \
              --model google/gemma-2-9b \
              --tensor-parallel-size 1
   ```
4. [ ] **Wait for Benchmarks:** The orchestrator will attempt to run the benchmarks against the provided URL once the engine is up.

### Option B: Manual Workflow

1. [ ] **Search and Rent:**
   ```bash
   python3 infra/vast_manager.py --search "RTX 3090"
   # Note the ID of the offer you want
   python3 infra/vast_manager.py --rent <OFFER_ID>
   ```
2. [ ] **Deploy Engine:** SSH into the instance and run the vLLM Docker container as shown above.
3. [ ] **Run Load Tester:**
   ```bash
   python3 bench/load_tester.py --url http://<instance-ip>:<port> --model gemma-2-9b --concurrency 10 --requests 50
   ```
4. [ ] **Cleanup:**
   ```bash
   python3 infra/vast_manager.py --destroy <INSTANCE_ID>
   ```

## 5. Interpreting Results

- [ ] **Locate Results:** The orchestrator saves results as `benchmark_<gpu_name>_<timestamp>.json`.
- [ ] **Key Metrics:**
  - **avg_ttft:** Time to First Token (ms). Lower is better for responsiveness.
  - **avg_itl:** Inter-Token Latency (ms). Lower is better for smooth reading experience.
  - **total_tps:** Aggregate Tokens Per Second across all concurrent users. Higher is better for throughput.
- [ ] **Analyze Efficiency:** Compare `total_tps` against the hourly cost of the instance to determine cost-effectiveness.

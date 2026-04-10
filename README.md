# Gemma Performance Lab on Vast.ai

This framework automates the benchmarking of Gemma models (and other LLMs) across different hardware configurations on Vast.ai. It measures critical performance metrics like Time to First Token (TTFT), Inter-Token Latency (ITL), and total throughput.

## Features
- **Automated Provisioning:** Uses the Vast.ai SDK to rent and manage GPU instances.
- **Asynchronous Load Testing:** Simulates concurrent users to measure real-world performance.
- **Metrics Collection:** Tracks TTFT, ITL, TPS, and cost efficiency.
- **Orchestration:** Manages the full lifecycle from search to teardown.

## Project Structure
- `CONCEPT.md`: Detailed architecture and objectives.
- `infra/vast_manager.py`: Infrastructure management logic.
- `bench/load_tester.py`: Load testing and metrics collection.
- `orchestrator.py`: End-to-end automation script.
- `requirements.txt`: Python dependencies.

## Getting Started

### Prerequisites
1. A Vast.ai account and API key.
2. Python 3.10+ installed.

### Installation
```bash
pip install -r requirements.txt
```

### Configuration
Set your Vast.ai API key:
```bash
vastai set api-key <YOUR_API_KEY>
```

### Usage

#### 1. Search for GPUs
```bash
python3 infra/vast_manager.py --search "RTX 4090"
```

#### 2. Run a Load Test
If you already have an OpenAI-compatible endpoint running:
```bash
python3 bench/load_tester.py --url http://<instance-ip>:<port> --model gemma-2-9b --concurrency 10 --requests 50
```

#### 3. Run the Full Orchestrator
```bash
python3 orchestrator.py --gpu "RTX_4090" --model "gemma-2-9b"
```

## Metrics Defined
- **TTFT:** Time to First Token (ms) - Measures responsiveness.
- **ITL:** Inter-Token Latency (ms) - Measures reading speed consistency.
- **TPS:** Tokens Per Second - Measures total throughput.
- **TPS/$:** Cost efficiency of the hardware.

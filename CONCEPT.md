# CONCEPT: Gemma Performance Lab on Vast.ai

## Overview
The Gemma Performance Lab is an automated benchmarking framework designed to evaluate the performance and cost-efficiency of Gemma models across the diverse hardware landscape of Vast.ai. By automating the entire lifecycle—from hardware provisioning to data analysis—it provides objective data to help developers choose the optimal GPU for their specific LLM workloads.

## Core Objectives
- **Hardware Benchmarking:** Compare high-end datacenter GPUs (A100, H100) against consumer-grade flagships (RTX 4090, 3090) for LLM serving.
- **Latency Analysis:** Quantify the user experience through Time to First Token (TTFT) and Inter-Token Latency (ITL).
- **Economic Evaluation:** Determine the "Tokens per Dollar" efficiency of different instance types.

## Architecture

### 1. Infrastructure Manager (`infra/vast_manager.py`)
- **Responsibility:** Automated lifecycle management of Vast.ai instances.
- **Features:**
    - Querying the Vast.ai marketplace for specific GPU models.
    - Automated renting of instances based on a "best-value" or "specific-match" policy.
    - SSH key management and instance readiness verification.
    - Guaranteed teardown to prevent runaway costs.

### 2. Load Test Engine (`bench/load_tester.py`)
- **Responsibility:** High-concurrency performance measurement.
- **Features:**
    - Asynchronous request handling using `aiohttp`.
    - Support for streaming LLM responses to capture granular timing data.
    - Configurable workloads (varying prompt lengths, output lengths, and concurrency).
    - Measurement of TTFT, ITL, and total tokens per second.

### 3. Orchestrator (`orchestrator.py`)
- **Responsibility:** End-to-end workflow automation.
- **Features:**
    - Sequencing: Provisioning -> Environment Setup -> Benchmarking -> Cleanup.
    - Deploying LLM engines (e.g., vLLM or Ollama) via Docker.
    - Aggregating raw data from multiple runs into a unified report.

## Target Metrics
- **TTFT (Time to First Token):** The time from request initiation to the first character received.
- **ITL (Inter-Token Latency):** Average time between successive tokens in a stream.
- **TPS (Tokens Per Second):** Total throughput across all concurrent users.
- **TPS/$:** Tokens per second divided by the hourly rental rate.

## Planned Workflow
1. **Search:** Identify available GPUs on Vast.ai that match the test requirements.
2. **Provision:** Rent the instance and wait for it to become reachable via SSH.
3. **Deploy:** Pull and run the serving engine Docker image with the target Gemma model.
4. **Benchmark:** Execute the load tester with a matrix of concurrency levels (e.g., 1, 10, 50, 100 users).
5. **Collect:** Extract logs and performance data.
6. **Teardown:** Immediately destroy the instance to stop billing.
7. **Report:** Generate comparative charts and tables.

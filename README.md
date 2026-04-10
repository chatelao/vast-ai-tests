# avast-ai-tests


`docker run --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN=<your_token> -p 8000:8000 vllm/vllm-openai:v0.4.0 --model google/gemma-4-31b-it --tensor-parallel-size 2

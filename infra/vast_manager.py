import argparse
import time
import sys
import os
import requests
import asyncio
import aiohttp
from vastai.sdk import VastAI
from vastai.utils import parse_env

class VastManager:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("VAST_AI_API_KEY")
        self.sdk = VastAI(api_key=self.api_key)

    def find_offers(self, gpu_name, num_gpus=1):
        query = f"gpu_name={gpu_name} num_gpus={num_gpus} rentable=True verified=True"
        try:
            offers = self.sdk.search_offers(query=query, order="dph_total")
            return offers
        except requests.exceptions.HTTPError as e:
            print(f"Error searching offers: {e}")
            if e.response is not None:
                print(f"Response: {e.response.text}")
            return []
        except Exception as e:
            print(f"Unexpected error searching offers: {e}")
            return []

    def rent_instance(self, offer_id, image="nvidia/cuda:12.1.1-devel-ubuntu22.04", disk=50, template_hash=None, env=None):
        if template_hash:
            print(f"Attempting to rent offer {offer_id} using template {template_hash}...")
            image = None
        else:
            print(f"Attempting to rent offer {offer_id} with image {image}...")

        # If env is a string, parse it into a dictionary as expected by the SDK/API
        if isinstance(env, str):
            env = parse_env(env)

        try:
            result = self.sdk.create_instance(id=offer_id, image=image, disk=disk, template_hash=template_hash, env=env)
        except requests.exceptions.HTTPError as e:
            print(f"Error renting instance: {e}")
            if e.response is not None:
                print(f"Response: {e.response.text}")
            return None
        except Exception as e:
            print(f"Unexpected error renting instance: {e}")
            return None

        if result.get("success"):
            instance_id = result.get("new_contract")
            print(f"Successfully created instance {instance_id}")
            return instance_id
        else:
            print(f"Failed to rent instance: {result}")
            return None

    def wait_for_ssh(self, instance_id, timeout=1200):
        print(f"Waiting for instance {instance_id} to be ready...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                instances = self.sdk.show_instances()
            except requests.exceptions.HTTPError as e:
                print(f"Error fetching instances: {e}")
                if e.response is not None:
                    print(f"Response: {e.response.text}")
                return None
            except Exception as e:
                print(f"Unexpected error fetching instances: {e}")
                return None

            instance = next((i for i in instances if i['id'] == instance_id), None)

            if instance:
                status = instance.get("status_msg") or instance.get("state")
                print(f"Current status: {status}")
                if instance.get("ssh_host") and instance.get("ssh_port"):
                    return instance

            time.sleep(15)
        print("Timeout waiting for instance")
        return None

    def destroy_instance(self, instance_id):
        print(f"Destroying instance {instance_id}...")
        try:
            return self.sdk.destroy_instance(id=instance_id)
        except requests.exceptions.HTTPError as e:
            print(f"Error destroying instance {instance_id}: {e}")
            if e.response is not None:
                print(f"Response: {e.response.text}")
            return None
        except Exception as e:
            print(f"Unexpected error destroying instance {instance_id}: {e}")
            return None

    def get_instance_details(self, instance_id):
        """Fetch current details for a specific instance using the REST API."""
        url = f"https://console.vast.ai/api/v0/instances/{instance_id}/"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            # The API might return {"instances": [...]}, {"instance": {...}}, or just the instance object
            if isinstance(data, dict):
                if "instances" in data:
                    instances = data["instances"]
                    if isinstance(instances, list):
                        return next((i for i in instances if str(i.get('id')) == str(instance_id)), None)
                    return instances
                if "instance" in data:
                    return data["instance"]
            return data
        except Exception as e:
            print(f"Error fetching instance details for {instance_id}: {e}")
            return None

    def get_current_instance_id(self):
        """Identifies the current Vast.ai instance ID by matching its public IP."""
        try:
            import urllib.request
            # Get public IP of the current machine
            public_ip = urllib.request.urlopen('https://api.ipify.org').read().decode('utf8')
            print(f"Detected public IP: {public_ip}")

            instances = self.sdk.show_instances()
            # Find instance that matches this public IP
            for inst in instances:
                if inst.get('public_ipaddr') == public_ip or inst.get('ssh_host') == public_ip:
                    print(f"Matched with instance ID: {inst['id']}")
                    return inst['id']
        except Exception as e:
            print(f"Error detecting current instance ID: {e}")
        return None

    def resolve_api_url(self, instance_id, internal_port=8000):
        """Resolves the external API URL for a given instance and internal port."""
        details = self.get_instance_details(instance_id)
        if not details:
            return None

        host = details.get('public_ipaddr', details.get('ssh_host'))
        port = internal_port
        ports = details.get('ports', {})
        if isinstance(ports, dict):
            for port_key, mappings in ports.items():
                if port_key.startswith(str(internal_port)):
                    if isinstance(mappings, list) and len(mappings) > 0:
                        mapping = mappings[0]
                        if isinstance(mapping, dict):
                            port = mapping.get('HostPort', port)
                    elif isinstance(mappings, (str, int)):
                        port = mappings
                    break

        return f"http://{host}:{port}"

    async def wait_for_api_ready(self, url, api_key=None, timeout=1200):
        """Polls the /v1/models endpoint until it returns 200 OK."""
        print(f"Waiting for LLM API to be ready at {url}...")
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < timeout:
                elapsed = int(time.time() - start_time)
                try:
                    async with session.get(f"{url}/v1/models", headers=headers) as response:
                        if response.status == 200:
                            print(f"API is ready after {elapsed}s!")
                            return True
                        else:
                            text = await response.text()
                            print(f"  ...API returned {response.status}: {text[:100]}")
                except Exception:
                    # Generic error (e.g. connection refused) is expected during startup
                    pass
                if elapsed % 30 == 0 and elapsed > 0:
                    print(f"  ...still waiting ({elapsed}s elapsed)")
                await asyncio.sleep(10)
        print("Timeout waiting for API to be ready.")
        return False

    def get_vllm_env_vars(self, model_name, api_key="vllm-benchmark-token"):
        """Generates the standard vLLM environment variable string."""
        hf_token = os.getenv("HF_TOKEN", "")
        vllm_args = "--dtype auto --enforce-eager --max-model-len 512 --block-size 16 --port 8000"
        return f"-e VLLM_MODEL={model_name} -e VLLM_ARGS='{vllm_args}' -e HF_TOKEN={hf_token} -e OPEN_BUTTON_TOKEN={api_key} -p 1111:1111 -p 7860:7860 -p 8000:8000 -p 8265:8265 -p 8080:8080"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vast.ai Instance Manager")
    parser.add_argument("--search", type=str, help="Search for GPUs by name")
    parser.add_argument("--rent", type=int, help="Rent an instance by offer ID")
    parser.add_argument("--destroy", type=int, help="Destroy an instance by ID")

    args = parser.parse_args()
    mgr = VastManager()

    if args.search:
        offers = mgr.find_offers(args.search)
        for o in offers[:5]:
            print(f"ID: {o['id']} | GPU: {o['gpu_name']} | Price: ${o['dph_total']}/hr")
    elif args.rent:
        mgr.rent_instance(args.rent)
    elif args.destroy:
        mgr.destroy_instance(args.destroy)
    else:
        parser.print_help()

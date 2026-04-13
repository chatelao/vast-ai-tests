import argparse
import time
import sys
import os
from vastai.sdk import VastAI

class VastManager:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("VAST_AI_API_KEY")
        self.sdk = VastAI(api_key=self.api_key)

    def find_offers(self, gpu_name, num_gpus=1):
        query = f"gpu_name={gpu_name} num_gpus={num_gpus} rentable=True verified=True"
        offers = self.sdk.search_offers(query=query, order="dph_total")
        return offers

    def rent_instance(self, offer_id, image="nvidia/cuda:12.1.1-devel-ubuntu22.04", disk=50, template_hash=None, env=None):
        if template_hash:
            print(f"Attempting to rent offer {offer_id} using template {template_hash}...")
            image = None
        else:
            print(f"Attempting to rent offer {offer_id} with image {image}...")
        result = self.sdk.create_instance(id=offer_id, image=image, disk=disk, template_hash=template_hash, env=env)
        if result.get("success"):
            instance_id = result.get("new_contract")
            print(f"Successfully created instance {instance_id}")
            return instance_id
        else:
            print(f"Failed to rent instance: {result}")
            return None

    def wait_for_ssh(self, instance_id, timeout=600):
        print(f"Waiting for instance {instance_id} to be ready...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            instances = self.sdk.show_instances()
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
        return self.sdk.destroy_instance(id=instance_id)

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

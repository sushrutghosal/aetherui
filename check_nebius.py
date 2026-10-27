import os
from openai import OpenAI

# read .env (same file the app uses)
if os.path.exists(".env"):
    for line in open(".env"):
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ.setdefault(k.replace("export ", "").strip(), v.strip().strip("'\""))

api_key = os.environ.get("NEBIUS_API_KEY")
if not api_key or "your-nebius" in api_key:
    raise ValueError("Put your Nebius key in .env (NEBIUS_API_KEY=...) first.")

# Tries the Token Factory URL first, then the older studio URL
for base_url in ["https://api.tokenfactory.nebius.com/v1/", "https://api.studio.nebius.com/v1/"]:
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        models = [m.id for m in client.models.list().data]
        print(f"\n[+] Connected to: {base_url}")
        nemotron_models = [m for m in models if "nemotron" in m.lower() or "vl" in m.lower() or "nvidia" in m.lower()]
        print("[+] Available NVIDIA / Vision Models on your account:")
        for m in nemotron_models:
            print(f"    -> {m}")
        break
    except Exception as e:
        print(f"[-] Failed {base_url}: {e}")
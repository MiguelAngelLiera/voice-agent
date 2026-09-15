import ssl
import os
from dotenv import load_dotenv
from mlx_lm import load, generate

# Load variables from .env file into os.environ
load_dotenv()

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Disable XET downloader
os.environ["HF_HUB_DISABLE_XET"] = "1"

print("Downloading/Loading model into Unified Memory...")
# mlx_lm automatically reads os.environ["HF_TOKEN"] under the hood
model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")

prompt = "In one sentence, explain how unified memory works."
response = generate(model, tokenizer, prompt=prompt, verbose=True)
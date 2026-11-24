<div align="center">

<img src="https://raw.githubusercontent.com/Studio1HQ/Studio1HQ/main/assets/banners/vision_model_jozu.webp" alt="Vision Language Model on Jozu RIC: Run a vision language model on a Jozu RIC and query it with a receipt image" width="100%" />

<br />
<br />

# Vision Language Model on Jozu RIC

### Run a vision language model on a Jozu RIC and query it with a receipt image

A notebook that packages Qwen3.6-35B-A3B and its multimodal projector as a ModelKit, serves them with a Jozu RIC and asks the model to describe a sample receipt.

<br />

[![Built by Studio1](https://img.shields.io/badge/Built_by_Studio1-f97316?style=for-the-badge&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxNjAgMTYwIiBmaWxsPSJub25lIj48cGF0aCBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGNsaXAtcnVsZT0iZXZlbm9kZCIgZD0iTTAgMzVDMCAxNi42ODA3IDE0LjA3NDQgMS42NDg0NCAzMiAwLjEyNjk1M1YxNDFINTEuMDUyM0M1Mi40MzA4IDEzNy4zNDQgNTQuNzQ0NiAxMzMuNjg5IDU3Ljk5MzggMTMwLjAzM0M2MS4xNDQ2IDEyNi41MDggNjQuNjg5MiAxMjMuNDQgNjguNjI3NyAxMjAuODI5QzcyLjU2NjIgMTE4LjM0OSA3Ni4zNTY5IDExNi43ODIgODAgMTE2LjEyOVY4Ni43NTQ0Qzc1LjQ3MDggODcuOTI5MiA3MS4zODQ2IDg5LjY5MTkgNjcuNzQxNSA5Mi4wNDE1QzY0IDk0LjUyMjUgNjAuNzUwOCA5Ny40NiA1Ny45OTM4IDEwMC44NTRDNTYuNTM0NSAxMDIuNzIgNTUuMjU0NSAxMDQuNjU5IDU0LjE1MzkgMTA2LjY3MVYwSDEyNUMxNDQuMzMgMCAxNjAgMTUuNjY5OSAxNjAgMzVWMTI1QzE2MCAxNDQuMzMgMTQ0LjMzIDE2MCAxMjUgMTYwSDEyNFYxOEgxMDQuOTQ4QzEwMy41NjkgMjEuNjgxNiAxMDEuMjU1IDI1LjM2MjggOTguMDA2MiAyOS4wNDQ0Qzk0Ljg1NTQgMzIuNTk0MiA5MS4zMTA4IDM1LjY4NDEgODcuMzcyMyAzOC4zMTRDODMuNDMzOCA0MC44MTIgNzkuNjQzMSA0Mi4zODk2IDc2IDQzLjA0NzRWNzIuNjMwNEM4MC41MjkyIDcxLjQ0NzMgODQuNjE1NCA2OS42NzI0IDg4LjI1ODUgNjcuMzA1N0M5MiA2NC44MDc2IDk1LjI0OTIgNjEuODQ5MSA5OC4wMDYyIDU4LjQzMDdDOTkuNDY1NSA1Ni41NTEzIDEwMC43NDUgNTQuNTk4NiAxMDEuODQ2IDUyLjU3MjNWMTYwSDM1QzE1LjY3IDE2MCAwIDE0NC4zMyAwIDEyNVYzNVoiIGZpbGw9IiNmZmZmZmYiLz48L3N2Zz4K)](https://studio1hq.com)
[![Built with Jozu](https://img.shields.io/badge/Built_with_Jozu-141414?style=for-the-badge)](https://jozu.com)

[Jozu docs](https://jozu.com/docs) &nbsp;·&nbsp; [Jozu](https://jozu.com) &nbsp;·&nbsp; [KitOps](https://kitops.org) &nbsp;·&nbsp; [More examples](https://github.com/Studio1HQ) &nbsp;·&nbsp; [studio1hq.com](https://studio1hq.com)

</div>

<br />

## Overview

This repo accompanies the blog post "Running Vision Language Models on Jozu Rapid Inference Container (RICs)". It contains the notebook (`main.ipynb`) and a sample `receipt.jpg`.

You download a 1-bit quantized Qwen3.6-35B-A3B and its `mmproj-BF16` multimodal projector, package both as a [KitOps](https://kitops.org) ModelKit, push to [Jozu Hub](https://jozu.com), run it on a Jozu Rapid Inference Container (RIC) and ask it to describe the receipt.

<p align="center"><img src="receipt.jpg" alt="Sample receipt used for the demo" width="40%" /></p>

## Features

- **Multimodal packaging**: The GGUF model is the Kitfile `model` and `mmproj-BF16.gguf` is a `parts` entry.
- **RIC with vision**: `llama-server` runs with `--mmproj` so the model accepts images.
- **OpenAI-compatible API**: The `openai` client calls `http://localhost:8000/v1`.
- **Visual question answering**: The sample script base64-encodes `receipt.jpg`, sends it with a prompt and prints the answer (`max_tokens=8192`).

## Tech stack

| Layer | Tool |
| --- | --- |
| Model | [Qwen3.6-35B-A3B GGUF (UD-IQ1_M)](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) and `mmproj-BF16.gguf` |
| Packaging | [KitOps](https://kitops.org) CLI and `kitops` Python library |
| Registry and runtime | [Jozu Hub](https://jozu.ml) and Jozu Rapid Inference Container |
| Client | [OpenAI Python SDK](https://github.com/openai/openai-python) |
| Runtime | Python 3.11+, Docker, Jupyter |

## Getting started

### Prerequisites

- Python 3.11 or later and Jupyter
- Docker
- A Jozu Hub account at [jozu.ml](https://jozu.ml)
- Homebrew on Linux or macOS (the notebook installs it if missing)
- A machine with an NVIDIA GPU and the NVIDIA container toolkit for the `--gpus all` Docker command

### Installation

```bash
git clone https://github.com/Studio1HQ/vision_model_jozu.git
cd vision_model_jozu
brew tap kitops-ml/kitops
brew install kitops
pip install openai kitops huggingface_hub
```

### Environment variables

No environment variables. The notebook authenticates with the email and password of your Jozu Hub account and uses placeholders you replace.

| Placeholder | Description |
| --- | --- |
| `email-prefix-here` | The part of your Jozu Hub email before the `@` (for `jack123@gmail.com` use `jack123`). Keep tags lowercase. |
| `full-email-here`, `password-here` | Jozu Hub credentials passed to `UserCredentials` in the notebook. |

### Run

Log in to Jozu Hub in a terminal (you are prompted for email and password), then open the notebook:

```bash
kit login jozu.ml
jupyter notebook main.ipynb
```


## How it works

1. Download both files from `unsloth/Qwen3.6-35B-A3B-GGUF` into `Qwen3.6-35B-A3B-GGUF/`.
2. Build the Kitfile with the projector under `parts` and push to `jozu.ml/email-prefix-here/qwen3.6-35b-a3b-gguf:latest`.
3. Launch the RIC:
   ```bash
   sudo docker run -it --rm -p 8000:8000 --gpus all \
     --entrypoint /app/llama-server \
     "jozu.ml/email-prefix-here/qwen3.6-35b-a3b-gguf/llama-cpp-cuda:latest" \
     -m /Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ1_M.gguf \
     --mmproj /Qwen3.6-35B-A3B-GGUF/mmproj-BF16.gguf \
     --jinja --chat-template-kwargs '{"enable_thinking": false}' \
     --host 0.0.0.0 --port 8000 -c 40965
   ```
4. Encode `receipt.jpg` as base64, send it through `client.chat.completions.create(...)` and print the response.

## Resources

- [Jozu documentation](https://jozu.com/docs): Jozu Hub and Rapid Inference Container guides.
- [Jozu](https://jozu.com): Product site.
- [KitOps documentation](https://kitops.org/docs/overview/): ModelKit and Kitfile reference.

<br />

---

<div align="center">

<a href="https://studio1hq.com">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Studio1HQ/website/main/brand/studio1-logo-on-dark.svg" />
    <img src="https://raw.githubusercontent.com/Studio1HQ/website/main/brand/studio1-logo-on-light.svg" alt="Studio1" width="160" />
  </picture>
</a>

Built by **[Studio1](https://studio1hq.com)**, the technical content and DevRel partner for devtool teams. Made with [Jozu](https://jozu.com).<br />
Browse more open-source examples at [github.com/Studio1HQ](https://github.com/Studio1HQ).

[![Website](https://img.shields.io/badge/studio1hq.com-0b0807?style=for-the-badge&logo=googlechrome&logoColor=white)](https://studio1hq.com)
[![Book a call](https://img.shields.io/badge/Book_a_strategy_call-f97316?style=for-the-badge&logo=googlecalendar&logoColor=white)](https://cal.com/studio1/collab)
[![X](https://img.shields.io/badge/%40Studio1HQ-141414?style=for-the-badge&logo=x&logoColor=white)](https://x.com/Studio1HQ)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0a66c2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/studio1hq/)

<sub><a href="https://studio1hq.com">studio1hq.com</a> · Technical content, docs, video and DevRel for devtools</sub>

</div>

# Blog Companion Repository

This repository accompanies the blog post **“Running Vision Language Models on Jozu Rapid Inference Container (RICs)”** and contains a receipt.jpg and Jupyter notebook (`main.ipynb`) that contains the comands and examples demonstrated in the article.

The goal of this repo is **reproducibility**: if you follow both the blog and notebook, you will be able to install the required tooling, pull the 1-bit quantized Qwen 3.6 and it's mmproj-BF16 (multimodal projector) from huggingface, package it as a ModelKit using KitOps, upload to Jozu Hub, run the model with Jozu Rapid Inference Container (RICs) and query it to describe the receipt's content.

---

## 📁 Repository Structure

```text
.
├── .gitignore
├── README.md
├── main.ipynb        # Commands & Examples shown on the blog
├── receipt.jpg
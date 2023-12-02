# LLM Inference

## Table of Contents

- [LLM Inference](#llm-inference)
  - [Table of Contents](#table-of-contents)
  - [Glossary](#glossary)
  - [Open Source Software](#open-source-software)
  - [Paper List](#paper-list)

## Glossary

* Prompt: the initial text or instruction given to the model.
* Prompt Phase (Prefill Phase): the phase to generate the first token based on the prompt.
* Generation Phase (Decoding Phase): genernate the next token based on the prompt and the previously generated tokens, in an **token-by-token** manner.
* Autoregressive: predicting **one** token at a time, conditioned on the previously generated tokens.
* KV (Key-Value) Cache: caching the attention Keys and Values in the Generation Phase, eliminating the recomputation for Keys and Values of previous tokens.
* Continuous Batching: as opposed to static batching (which batches requests together and starts processing only when all requests within the batch are ready), continuously batches requests and maximizes memory utilization.
* Offloading: transfering data between GPU memory and main memory or NVMe storage, as GPU memory is limited.
* Post-training quantization: quantizing the weights and activations of the model **after** the model has been trained.
* Quantization-Aware Training: incorporating quantization considerations during training.

## Open Source Software

| Name | Stars | Hardware | Org |
| --- | --- | ---| --- |
| [Transformers](https://github.com/huggingface/transformers) | ![](https://img.shields.io/github/stars/huggingface/transformers.svg?style=social) | CPU / NVIDIA GPU / TPU / AMD GPU | Hugging Face |
| [Text Generation Inference](https://github.com/huggingface/text-generation-inference) | ![](https://img.shields.io/github/stars/huggingface/text-generation-inference.svg?style=social) | CPU / NVIDIA GPU / AMD GPU | Hugging Face |
| [gpt-fast](https://github.com/pytorch-labs/gpt-fast) | ![](https://img.shields.io/github/stars/pytorch-labs/gpt-fast.svg?style=social) | CPU / NVIDIA GPU / AMD GPU | PyTorch |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | ![](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM.svg?style=social) | NVIDIA GPU | NVIDIA |
| [vLLM](https://github.com/vllm-project/vllm) | ![](https://img.shields.io/github/stars/vllm-project/vllm.svg?style=social) | NVIDIA GPU | UC Berkeley |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) / [ggml](https://github.com/ggerganov/ggml) | ![](https://img.shields.io/github/stars/ggerganov/llama.cpp.svg?style=social)![](https://img.shields.io/github/stars/ggerganov/ggml.svg?style=social) | CPU / Apple Silicon / NVIDIA GPU / AMD GPU | ggml |
| [ctransformers](https://github.com/marella/ctransformers) | ![](https://img.shields.io/github/stars/marella/ctransformers.svg?style=social) | CPU / Apple Silicon / NVIDIA GPU / AMD GPU | Ravindra Marella |
| [DeepSpeed](https://github.com/microsoft/DeepSpeed) | ![](https://img.shields.io/github/stars/microsoft/DeepSpeed.svg?style=social) | CPU / NVIDIA GPU | Microsoft |
| [FastChat](https://github.com/lm-sys/FastChat) | ![](https://img.shields.io/github/stars/lm-sys/FastChat.svg?style=social) | CPU / NVIDIA GPU / Apple Silicon | lmsys.org |
| [MLC-LLM](https://github.com/mlc-ai/mlc-llm) | ![](https://img.shields.io/github/stars/mlc-ai/mlc-llm.svg?style=social) | CPU / NVIDIA GPU | MLC |
| [LightLLM](https://github.com/ModelTC/lightllm) | ![](https://img.shields.io/github/stars/ModelTC/lightllm.svg?style=social) | CPU / NVIDIA GPU | SenseTime |
| [LMDeploy](https://github.com/InternLM/lmdeploy) | ![](https://img.shields.io/github/stars/InternLM/lmdeploy.svg?style=social) | CPU / NVIDIA GPU | Shanghai AI Lab & SenseTime |
| [OpenLLM](https://github.com/bentoml/OpenLLM) | ![](https://img.shields.io/github/stars/bentoml/OpenLLM.svg?style=social) | CPU / NVIDIA GPU / AMD GPU | BentoML |
| [OpenPPL.nn](https://github.com/openppl-public/ppl.nn) / [OpenPPL.nn.llm](https://github.com/openppl-public/ppl.nn.llm) | ![](https://img.shields.io/github/stars/openppl-public/ppl.nn.svg?style=social)![](https://img.shields.io/github/stars/openppl-public/ppl.nn.llm.svg?style=social) | CPU / NVIDIA GPU | OpenMMLab & SenseTime |
| [ScaleLLM](https://github.com/vectorch-ai/ScaleLLM) | ![](https://img.shields.io/github/stars/vectorch-ai/ScaleLLM?style=social) | NVIDIA GPU | Vectorch |
| [RayLLM](https://github.com/ray-project/ray-llm) | ![](https://img.shields.io/github/stars/ray-project/ray-llm?style=social) | CPU / NVIDIA GPU / AMD GPU | Anyscale |
| [Xorbits Inference](https://github.com/xorbitsai/inference) | ![](https://img.shields.io/github/stars/xorbitsai/inference.svg?style=social) | CPU / NVIDIA GPU / AMD GPU | Xorbits |

## Paper List

| Name                 | Paper Title                                                                              | Paper Link                                                                                                                     | Artifact                                             | Keywords                                     | Recommend  |
| -------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------- | -------------------------------------------- | ---------- |
| LLaMA                | LLaMA: Open and Efficient Foundation Language Models                                     | [arXiv 23](https://arxiv.org/pdf/2302.13971.pdf)                                                                               | [Code](https://github.com/facebookresearch/llama)    | Pre-training                                 | ⭐️⭐️⭐️⭐️⭐️ |
| Llama 2              | Llama 2: Open Foundation and Fine-Tuned Chat Models                                      | [arXiv 23](https://arxiv.org/pdf/2307.09288.pdf)                                                                               | [Model](https://huggingface.co/meta-llama)           | Pre-training / Fine-tuning / Safety          | ⭐️⭐️⭐️⭐️   |
| Multi-Query          | Fast Transformer Decoding: One Write-Head is All You Need                                | [arXiv 19](https://arxiv.org/pdf/1911.02150.pdf)                                                                               |                                                      | Architecture                                 | ⭐️⭐️⭐️     |
| Grouped-Query        | GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints     | [arXiv 23](https://arxiv.org/pdf/2305.13245.pdf)                                                                               |                                                      | Architecture                                 | ⭐️⭐️⭐️     |
| RoPE | Roformer: Enhanced transformer with rotary position embedding                            | [arXiv 21](https://arxiv.org/pdf/2104.09864.pdf)                                                                               |                                                      | Position Encoding                            | ⭐️⭐️⭐️⭐️   |
| Megatron-LM | Efficient large-scale language model training on GPU clusters using megatron-LM          | [SC 21](https://dl.acm.org/doi/pdf/10.1145/3458817.3476209)                                                                    | [Code](https://github.com/NVIDIA/Megatron-LM/)       | Parallelism                                  | ⭐️⭐️⭐️⭐️⭐️ |
| Google's Practice | Efficiently Scaling Transformer Inference                                                | [MLSys 23](https://proceedings.mlsys.org/paper_files/paper/2023/file/523f87e9d08e6071a3bbd150e6da40fb-Paper-mlsys2023.pdf)     |                                                      | Parallelism                                  | ⭐️⭐️⭐️⭐️   |
| FlashAttention | Fast and Memory-Efficient Exact Attention with IO-Awareness                              | [NeurIPS 23](https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf) | [Code](https://github.com/Dao-AILab/flash-attention) | Effiencent Attention / GPU                   | ⭐️⭐️⭐️⭐️⭐️ |
| Orca | Orca: A distributed serving system for Transformer-Based generative models               | [OSDI 22](https://www.usenix.org/system/files/osdi22-yu.pdf)                                                                   | [Code](https://github.com/vllm-project/vllm)         | Continuous Batching                          | ⭐️⭐️⭐️⭐️⭐️ |
| PagedAttention | Efficient Memory Management for Large Language Model Serving with PagedAttention         | [SOSP 23](https://dl.acm.org/doi/pdf/10.1145/3600006.3613165)                                                                  | [Code](https://github.com/vllm-project/vllm)         | Effiencent Attention / Continuous Batching / | ⭐️⭐️⭐️⭐️⭐️ |
| FlexGen              | FlexGen: High-throughput generative inference of large language models with a single GPU | [ICML 23](https://proceedings.mlr.press/v202/sheng23a/sheng23a.pdf)                                                            | [Code](https://github.com/FMInference/FlexGen)       | Offloading                                   | ⭐️⭐️⭐️     |
| Speculative Decoding | Fast Inference from Transformers via Speculative Decoding                                | [ICML 23](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf)                                                    |                                                      | Sampling                                     | ⭐️⭐️⭐️⭐️   |
| LLM.int8()           | LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale                        | [NeurIPS 22](https://proceedings.neurips.cc/paper_files/paper/2022/file/c3ba4962c05c49636d4c6206a97e9c8a-Paper-Conference.pdf) | [Code](https://github.com/timdettmers/bitsandbytes)  | Quantization |⭐️⭐️⭐️⭐️ |
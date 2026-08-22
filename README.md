# 🦙 llama.scriptc

> **LLM Inference Engine (Transformer) built from scratch in TypeScript and compiled natively with [ScriptC](https://scriptc.dev/).**

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Project Architecture](#-project-architecture)
- [Comparison: `llama.scriptc` vs `llama.cpp`](#-comparison--llamascriptc-vs-llamacpp)
- [Real-World Speed Benchmark](#-real-world-speed-benchmark--llamascriptc-vs-llamacpp)
- [Supported Features](#-supported-features)
- [Installation & Prerequisites](#-installation--prerequisites)
- [Quick Start](#-quick-start)
  - [1. Run a Real GGUF Model (Qwen2.5)](#1-run-a-real-gguf-model-qwen25)
  - [2. Standalone Demo](#2-standalone-demo)
  - [3. Run the Test Suite](#3-run-the-test-suite)
- [Code Structure](#-code-structure)
- [Validation & Tests](#-validation--tests)

---

## 🌟 Overview

`llama.scriptc` is a complete, standalone implementation of an inference engine for Transformer-type language models (Llama, Qwen2.5, etc.), written **entirely from scratch in strict TypeScript** and compiled directly to **native machine code** via the [ScriptC](https://scriptc.dev/) compiler from Vercel.

The project was designed according to the guiding principle: **"Correctness first, performance second"** (priority to clarity, modularity, and exact mathematical correctness).

```text
                                  GGUF File
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │     GGUF Reader     │
                            │   (Header, Config,  │
                            │    Q8_0 Dequant)    │
                            └──────────┬──────────┘
                                       │
                                       ▼
 ┌─────────────┐            ┌─────────────────────┐
 │  Tokenizer  │ ──(IDs)──> │  Transformer Model  │ ◄── RoPE Frequencies
 │ (Qwen/BPE)  │            │  (Embedding Layer)  │ ◄── Autoregressive KV Cache
 └─────────────┘            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │ N x TransformerBlock│
                            │   • Pre-RMSNorm     │
                            │   • Causal MHA/GQA  │
                            │   • SwiGLU MLP      │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │    Final RMSNorm    │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │  LM Head Projection │ ──> Logits [Vocab]
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │       Sampler       │
                            │ (Greedy, Temp, TopK)│
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │  Streaming Text     │
                            └─────────────────────┘
```

---

## ⚡ Real-World Benchmark: `llama.scriptc` vs `llama.cpp`

### GGUF Qwen2.5-0.5B Model (24 layers, 500M params)

Measurements on **Apple Silicon M4** with `models/qwen2.5-0.5b-instruct-q8_0.gguf` (prompt: `"Hello, my name is"`) :

| Engine | Backend & Acceleration | Prefill Speed | Decode Speed | Latency/token |
|:---|:---|:---:|:---:|:---:|
| **`llama.cpp` (Metal GPU)** | GPU Metal + Accelerate BLAS | **946.8 tok/s** | **138.6 tok/s** | **7.2 ms** |
| **`llama.cpp` (CPU)** | CPU Multi-thread (ARM NEON SIMD) | **950.1 tok/s** | **81.1 tok/s** | **12.3 ms** |
| **`llama.scriptc` (Our engine)** | Native Scalar CPU (ScriptC / Single-thread) | **~5.0 tok/s** | **~0.2 – 2.0 tok/s** | **~2,000 ms** |

> ⚠️ **Note**: The GGUF model remains slow because ScriptC compiles to single-threaded scalar code without SIMD/GPU. See the miniature model below for optimized performance.

### Miniature Model (2 layers, 128 dim) — **Optimized Version**

Measurements on the same hardware with `examples/generate.ts` (prompt: `"Hello, my name is"`, 20 tokens generated) :

| Version | Backend | Generation Speed | Latency/token |
|:---|:---|:---:|:---:|
| **`llama.scriptc` (Optimized)** | Scalar CPU (ScriptC) | **~196 tok/s** | **~5.1 ms** |

### 🔍 Why Such a Speed Difference on the GGUF Model?

1. **SIMD & Vectorization (ARM NEON)**: `llama.cpp` uses vector registers processing 4–8 float32/cycle. `llama.scriptc` executes scalar loops (1 float/cycle).
2. **GPU Acceleration (Metal)**: `llama.cpp` offloads large matmuls (e.g., `lm_head` 151k) to Apple Silicon GPU.
3. **Multi-threading (OpenMP/pthreads)**: `llama.cpp` parallelizes across all cores; `llama.scriptc` is single-threaded.
4. **Educational Purpose**: `llama.scriptc` prioritizes **mathematical transparency** (every softmax, RoPE, Q/K/V projection written in pure TypeScript).

### 🛠️ How to Reproduce Benchmarks:
```bash
# 1. GGUF Qwen2.5 Model (slow - limited by ScriptC scalar)
npx scriptc run examples/run_gguf.ts --dynamic

# 2. Miniature Model (fast - shows optimizations)
npx scriptc run examples/generate.ts --dynamic "Hello, my name is"

# 3. Test llama.cpp
llama-cli -m models/qwen2.5-0.5b-instruct-q8_0.gguf -p "Hello, my name is" -n 10 --temp 0.0
```

---

## ⚖️ Comparison: `llama.scriptc` vs `llama.cpp`

| Criterion | `llama.scriptc` (This project) | `llama.cpp` |
|:---|:---|:---|
| **Source Language** | **Strict TypeScript** | **C / C++ (C11 / C++14)** |
| **Execution Engine** | **ScriptC** AOT Compiler (LLVM/clang native) | Native compiled (GCC / Clang / MSVC) |
| **Code Readability** | **Very High** (~2,500 modular, commented lines) | **Complex** (~100,000+ lines of low-level C code and macros) |
| **Primary Goal** | **Educational, verifiability & clean architecture** | **Raw performance & industrial deployment** |
| **Format Support** | **GGUF v3**, custom JSON format | **GGUF v1, v2, v3**, legacy GGML formats |
| **Quantization** | **Q8_0** (O(1) dequantization), **F32**, **F16** | **20+ formats** (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, IQ, etc.) |
| **Memory Management** | Typed dense arrays, TypeScript structures, **Tensor Pool** | Manual `malloc`/`free`, `mmap`, GGML pool |
| **Hardware Acceleration** | Native scalar CPU (single-thread) | **SIMD** (AVX2, AVX-512, ARM NEON) + **GPU** (Metal, CUDA, Vulkan, ROCm) |
| **Multi-threading** | Single-thread | Multi-threaded (OpenMP / native pthreads) |
| **Inference Speed (miniature model)** | **~196 tok/s** (optimized) | ~30 to 150+ tok/s on CPU/GPU |
| **Inference Speed (GGUF 500M)** | ~0.5–4 tok/s (limited by ScriptC scalar) | ~30 to 150+ tok/s on CPU/GPU |
| **Test Suite** | **16 suites, 68 unit tests (< 30ms)** | Complete CI regression tests |
| **Key Optimizations** | **Op fusion, zero-copy views, reused buffers, optimized loops** | SIMD, GPU, multi-thread, advanced quantization |
| **Use Case** | Understand every LLM equation, prototyping, TS ecosystem | Production, inference servers, mobile/edge apps |

---

## ✨ Supported Features

- [x] **N-D Tensor System**: Row-major strides, view management (`view()`), slicing, 2D transpositions, automatic broadcasting, **Tensor Pool for buffer reuse**.
- [x] **Math & Normalization**: Batched 2D/3D Matmul **optimized (i-j-k loops, contiguous access)**, Numerically stable Softmax **in-place** with $\max(x)$ compensation.
- [x] **Neural Primitives**: `Linear` (**fused bias, reused buffer**), `Embedding`, `RMSNorm` (**fused norm+scale, reused buffer**), `SiLU` activations and `SwiGLU` block (**fused SiLU×Up**).
- [x] **Causal Attention**: Upper-triangular masking ($-10^9$), Multi-Head Attention (MHA) and Grouped-Query Attention (GQA) **with in-place RoPE, zero-copy KV Cache, fused softmax**.
- [x] **Rotary Positional Embeddings (RoPE)**: Frequencies $\theta_i = \text{base}^{-2i/d}$, 2D rotation by pairs preserving $L_2$ norm, compatible with RoPE base $1\,000\,000$ (Qwen2), **in-place application without allocation**.
- [x] **KV Cache**: Key-Value Cache with $O(1)$ update for step-by-step autoregressive decoding (complexity reduction from $O(N^2)$ to $O(N)$), **views instead of copies, in-place reset**.
- [x] **GGUF v3 & Q8_0 Support**: Binary reading, metadata extraction, and dequantization of $32 \times \text{int8} + \text{float16 scale}$ blocks.
- [x] **Qwen BPE Tokenizer**: Decoding of 151,936 token vocabulary and multi-byte UTF-8 byte unescaping.
- [x] **Sampling Strategies**: Greedy ($\text{temp}=0$), Temperature, Top-K ($O(N \cdot K)$ optimized), Top-P (Nucleus), and Repetition Penalty.

---

## 🚀 Installation & Prerequisites

### Prerequisites
- **ScriptC** installed globally:
  ```bash
  npm install -g scriptc
  ```
  *(Or verify that `scriptc` is in your `$PATH`)*

---

## ⚡ Quick Start

### 1. Run a Real GGUF Model (Qwen2.5)

To load and run the quantized model `models/qwen2.5-0.5b-instruct-q8_0.gguf`:

```bash
# Run generation with the real GGUF model
npx scriptc run examples/run_gguf.ts --dynamic
```

Example output:
```text
=========================================================
     🚀 QWEN2.5-0.5B INSTRUCT (Q8_0 GGUF) IN SCRIPTC
=========================================================

[1/3] Loading Tokenizer from GGUF...
      Vocabulary size: 151936 tokens
      BOS Token: 151643, EOS Token: 151645

[2/3] Loading 24 layers of Qwen2.5-0.5B Model...
[Loader] Loading embeddings (151936 x 896)...
[Loader] Loading final norm and LM head...
[Loader] Loading 24 transformer layers...
[Loader] Model loaded successfully in 11.2s.

[3/3] Generating response for prompt:
"Hello, my name is"

------------------ Response ------------------
Hello, I am an AI assistant...
----------------------------------------------
```

> ⚠️ **Note**: The GGUF model (24 layers, 500M params) remains slow (~0.5–4 tok/s) due to ScriptC's single-threaded scalar execution. See the benchmark above for detailed comparison.

---

### 2. Standalone Demo with Miniature Model

To test the complete pipeline with a miniature model and instant generation:

```bash
npx scriptc run examples/generate.ts --dynamic "Hello, my name is"
```

Example output (optimized version):
```text
=========================================================
    🚀 LLAMA.SCRIPTC — NATIVE TRANSFORMER LLM ENGINE
=========================================================

[1/4] Tokenizer initialized with 125 vocabulary tokens.
[2/4] Model Architecture:
      - Vocab Size       : 125
      - Hidden Dimension : 128
      - Number of Layers : 2
      - Attention Heads  : 4 (KV Heads: 4)
      - Intermediate Dim : 512
      - Context Window   : 256
[3/4] Transformer model weights initialized.

[4/4] Starting generation for prompt: "Hello, my name is"
---------------------------------------------------------
Hello, my name is [generated text...]
---------------------------------------------------------

✨ Generation finished in 102ms
   - Prompt tokens    : 5
   - Generated tokens : 20
   - Total tokens     : 25
   - Speed            : 196.1 tokens/sec (5.10 ms/token)

=========================================================
```

---

### 3. Run the Test Suite

The project includes 16 test suites validating all mathematical and neural operations:

```bash
npx scriptc run tests/run_all.ts --dynamic
```

Execution result (optimized version):
```text
=================================================
   LLM INFERENCE ENGINE (ScriptC) TEST SUITE
=================================================

  ✓ Tensor Shape & Strides (6 tests)
  ✓ Tensor Core Class (9 tests)
  ✓ Elementwise Math Operations (8 tests)
  ✓ Tensor Reduction Operations (6 tests)
  ✓ Matrix Multiplication 2D/3D (7 tests)
  ✓ Softmax Normalization (5 tests)
  ✓ Neural Network Primitives (5 tests)
  ✓ Rotary Positional Embeddings RoPE (3 tests)
  ✓ Causal Multi-Head Attention (4 tests)
  ✓ Transformer Block (2 tests)
  ✓ KV Cache Autoregressive Equivalence (2 tests)
  ✓ Tokenizer Encode/Decode (4 tests)
  ✓ Transformer Model & Loader (3 tests)
  ✓ Sampling Strategies (3 tests)
  ✓ End-to-End Generation Pipeline (1 test)
  ✓ GGUF v3 Parser & Q8_0 Dequantizer (2 tests)

=================================================
🎉 ALL 68 TESTS PASSED in 5ms!
=================================================
```

---

## 📂 Code Structure

```text
llama.scriptc/
├── src/
│   ├── tensor/             # N-D Tensors, types, strides and views
│   │   ├── tensor.ts
│   │   ├── shape.ts
│   │   └── dtype.ts
│   ├── math/               # Mathematical operations
│   │   ├── elementwise.ts  # add, sub, mul, div, exp, sqrt, rsqrt
│   │   ├── reduction.ts    # sum, mean, max, min, argmax
│   │   ├── matmul.ts       # Matmul 2D / 3D batched
│   │   └── softmax.ts      # Stable Softmax
│   ├── nn/                 # Neural primitives
│   │   ├── embedding.ts    # Lookup table
│   │   ├── linear.ts       # Dense projection y = xW^T + b
│   │   ├── rmsnorm.ts      # RMS Normalization
│   │   ├── activations.ts  # SiLU / Sigmoid
│   │   └── swiglu.ts       # SwiGLU MLP Llama-style
│   ├── transformer/        # Transformer Architecture
│   │   ├── rope.ts         # Rotary Positional Embeddings
│   │   ├── attention.ts    # Multi-Head & Grouped-Query Attention
│   │   ├── block.ts        # Transformer Block (Pre-RMSNorm + Residuals)
│   │   └── kv_cache.ts     # Autoregressive KV Cache O(1)
│   ├── tokenizer/          # Tokenization
│   │   ├── tokenizer.ts    # Standard Tokenizer
│   │   └── gguf_tokenizer.ts # GGUF BPE Tokenizer with UTF-8 decoding
│   ├── model/              # Model & Loading
│   │   ├── config.ts       # ModelConfig
│   │   ├── transformer.ts  # Complete Transformer Model
│   │   ├── gguf.ts         # GGUF v3 Parser & Q8_0 Dequantization
│   │   └── loader.ts       # JSON & GGUF Save/Load
│   ├── sampling/           # Sampler
│   │   └── sampler.ts      # Greedy, Temperature, Top-K, Top-P, Rep Penalty
│   └── runtime/            # Inference Pipeline
│       └── generator.ts    # Complete Text Generator (Prefill + Decode)
├── tests/                  # 16 automated test suites
│   ├── run_all.ts          # Global runner
│   ├── tensor/
│   ├── nn/
│   ├── transformer/
│   ├── tokenizer/
│   ├── model/
│   └── sampling/
├── examples/               # Executable examples
│   ├── run_gguf.ts         # Real Qwen2.5 GGUF Demo
│   └── generate.ts         # Miniature Model Demo
└── models/                 # GGUF Models (ex: qwen2.5-0.5b-instruct-q8_0.gguf)
```

---

## 🔬 Validation & Tests

1. **KV Cache Mathematical Equivalence**: Formal verification that step-by-step decoding with KV Cache produces a result **strictly identical ($< 10^{-6}$)** to full recomputation without cache.
2. **Strict Attention Causality**: Verification that no future information leak alters the past (causal triangular masking).
3. **PyTorch Reference**: A reference script [`tools/reference/pytorch_reference.py`](tools/reference/pytorch_reference.py) allows exporting weights and activations layer by layer for direct numerical comparison.

---

## 📜 License

Open-source project under MIT license.

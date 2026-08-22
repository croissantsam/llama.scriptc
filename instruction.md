# Instructions --- Build an LLM Inference Engine in ScriptC

## 1. Objective

Build **from scratch** an inference engine for Transformer-type language
models in ScriptC.

The project must prioritize:

-   complete understanding of how an LLM works;
-   a simple and modular architecture;
-   numerical correctness before performance;
-   unit tests and reference tests;
-   progressive optimization;
-   no dependency on an existing inference engine for the
    core runtime.

The first objective is not to beat llama.cpp in performance. The
first objective is to obtain a **correct, readable, and verifiable**
implementation capable of generating text.

------------------------------------------------------------------------

## 2. Development Principle

Always proceed in this order:

1.  Understand the component.
2.  Define its data structures.
3.  Implement a simple version.
4.  Write tests.
5.  Compare results with a reference.
6.  Measure performance.
7.  Optimize only after validation.

Never optimize an operation whose correctness has not been
demonstrated.

------------------------------------------------------------------------

## 3. Target Architecture

The engine must progressively evolve toward this architecture:

``` text
                    Model File
                       |
                       v
                +--------------+
                | Model Loader |
                +------+-------+
                       |
                       v
                +--------------+
                |   Tensors    |
                +------+-------+
                       |
                       v
             +-------------------+
             | Transformer Model |
             +---------+---------+
                       |
             +---------+---------+
             |                   |
             v                   v
        KV Cache              Logits
                                 |
                                 v
                            +---------+
                            | Sampler |
                            +----+----+
                                 |
                                 v
                              Token ID
                                 |
                                 v
                              Tokenizer
                                 |
                                 v
                               Text
```

------------------------------------------------------------------------

## 4. Phase 0 --- Understand ScriptC

Before implementing the LLM engine, precisely document the capabilities
of ScriptC.

To determine:

-   type system;
-   pointers;
-   references;
-   memory allocation;
-   arrays;
-   structures;
-   modules;
-   functions;
-   error handling;
-   files;
-   I/O;
-   numeric conversions;
-   `float32`;
-   `float16` / `bfloat16` if available;
-   threads;
-   SIMD;
-   native access;
-   FFI;
-   potential GPU;
-   build system;
-   test system;
-   profiling/debugging.

Create a small test program for each required capability.

**Do not assume a feature exists. Verify it.**

------------------------------------------------------------------------

## 5. Phase 1 --- Tensor System

Create a `Tensor` type.

Minimum recommended:

``` text
Tensor
├── data
├── dtype
├── shape
├── stride
└── size
```

The tensor must support at minimum:

``` text
create
free
reshape
view
get
set
fill
copy
```

Then implement:

``` text
add
sub
mul
div
sum
mean
max
exp
sqrt
rsqrt
transpose
matmul
softmax
```

Start exclusively with `float32`.

### Constraints

-   avoid unnecessary copies;
-   clearly distinguish ownership and views;
-   verify dimensions;
-   correctly handle strides;
-   test edge cases.

------------------------------------------------------------------------

## 6. Phase 2 --- Numerical Tests

Every Tensor operation must have tests.

Example:

``` text
A = [[1, 2],
     [3, 4]]

B = [[5, 6],
     [7, 8]]

C = A @ B
```

Expected result:

``` text
[[19, 22],
 [43, 50]]
```

Tests must also cover:

-   1D tensors;
-   2D tensors;
-   3D tensors;
-   incompatible dimensions;
-   empty tensors if the runtime allows them;
-   negative values;
-   large values;
-   small values;
-   NaN/Inf when relevant.

------------------------------------------------------------------------

## 7. Phase 3 --- Neural Network Primitives

Implement:

``` text
Linear
Embedding
RMSNorm
Softmax
SwiGLU
```

### Linear

``` text
y = xW + b
```

### RMSNorm

``` text
RMS(x) = sqrt(mean(x²) + epsilon)

y = x / RMS(x) * weight
```

### Softmax

Use a numerically stable implementation:

``` text
softmax(x_i) =
    exp(x_i - max(x))
    ------------------
    sum(exp(x_j - max(x)))
```

### SwiGLU

Implement the variant used by Llama-like architectures.

------------------------------------------------------------------------

## 8. Phase 4 --- Attention

First implement simple causal attention.

``` text
Q = XWq
K = XWk
V = XWv

Attention(Q,K,V) =
    softmax(QKᵀ / sqrt(d))V
```

Then add:

-   causal mask;
-   multi-head attention;
-   RoPE;
-   grouped-query attention if necessary.

Before any optimization, verify results on very small
tensors.

------------------------------------------------------------------------

## 9. Phase 5 --- RoPE

Implement Rotary Positional Embeddings.

Verify:

-   frequency;
-   position;
-   dimension;
-   dimension pair;
-   behavior at position 0;
-   consistency with a reference implementation.

Create independent numerical tests.

------------------------------------------------------------------------

## 10. Phase 6 --- Transformer Block

Build:

``` text
Input
  |
  v
RMSNorm
  |
  v
Attention
  |
  +------ residual
  |
  v
RMSNorm
  |
  v
SwiGLU / MLP
  |
  +------ residual
  |
  v
Output
```

The Transformer must be composed of independent modules.

Avoid putting all logic in a single function.

------------------------------------------------------------------------

## 11. Phase 7 --- KV Cache

Implement the KV cache before aiming for efficient generation.

Objective:

``` text
Prompt
  |
  +--> Prefill
  |
  +--> KV Cache
           |
           v
       Decode token
           |
           v
       KV Cache update
           |
           v
       Next token
```

The cache must avoid recalculating historical keys and values at
each token.

Document precisely:

-   memory layout;
-   dimensions;
-   capacity;
-   current position;
-   long context handling;
-   allocation;
-   reuse.

------------------------------------------------------------------------

## 12. Phase 8 --- Tokenizer

Implement or integrate a tokenizer only after clearly defining
the required format.

The tokenizer must provide:

``` text
encode(text) -> token_ids

decode(token_ids) -> text
```

Start with a simple format.

Then support the tokenizers needed for target models.

------------------------------------------------------------------------

## 13. Phase 9 --- Model Loader

Create an abstraction:

``` text
ModelLoader
├── metadata
├── tensors
└── model configuration
```

The loader must retrieve:

-   vocab size;
-   hidden size;
-   number of layers;
-   number of heads;
-   number of KV heads;
-   head dimension;
-   context;
-   RoPE parameters;
-   dtype;
-   weights.

Start with **a simple, project-controlled file format**.

Add GGUF or other formats later.

Do not start by supporting all existing formats.

------------------------------------------------------------------------

## 14. Phase 10 --- First Model

The first target model must be intentionally small.

Example:

``` text
vocab       = 1000
hidden      = 128
layers      = 2
heads       = 4
kv_heads    = 4
intermediate = 512
```

The goal is to be able to do:

``` text
prompt
  ↓
tokenizer
  ↓
Transformer
  ↓
logits
  ↓
sampler
  ↓
token
  ↓
decode
  ↓
text
```

------------------------------------------------------------------------

## 15. Phase 11 --- Validation with PyTorch

Create a small reference model in Python/PyTorch.

For each step compare:

``` text
ScriptC output
vs
PyTorch output
```

Compare in particular:

-   embeddings;
-   RMSNorm;
-   Linear;
-   RoPE;
-   Q/K/V;
-   attention;
-   MLP;
-   Transformer block;
-   logits.

Use an explicit numerical tolerance.

Example:

``` text
abs(a - b) < epsilon
```

and, when necessary:

``` text
relative_error(a, b) < epsilon
```

------------------------------------------------------------------------

## 16. Phase 12 --- Sampling

Progressively implement:

``` text
greedy
temperature
top-k
top-p
repetition penalty
```

The sampler must be independent of the Transformer.

Conceptual interface:

``` text
token = sampler.sample(logits)
```

------------------------------------------------------------------------

## 17. Phase 13 --- Generation

Create a minimal API:

``` text
generate(
    prompt,
    max_tokens,
    temperature,
    top_k,
    top_p
)
```

Usage example:

``` text
generate(
    "Hello, my name is",
    100,
    0.7,
    40,
    0.9
)
```

------------------------------------------------------------------------

## 18. Phase 14 --- Performance

Only start optimization after complete validation.

Measure separately:

``` text
Model loading
Prefill
Decode
Tokens/sec
Latency/token
Memory usage
Peak memory
```

Identify hotspots with a profiler.

First optimizations must target:

1.  `matmul`
2.  attention
3.  memory copies
4.  KV cache
5.  allocation
6.  element-wise operations

------------------------------------------------------------------------

## 19. Phase 15 --- SIMD

Once the naive version is correct:

``` text
Scalar
  ↓
SIMD
  ↓
Multithreading
  ↓
Optimized kernels
```

Determine the possibilities offered by ScriptC:

``` text
AVX2
AVX-512
NEON
SVE
```

depending on the platform.

Never break the reference implementation during an optimization.

Keep a simple version as a test oracle.

------------------------------------------------------------------------

## 20. Phase 16 --- Quantization

Progressively add:

``` text
FP32
 ↓
FP16/BF16
 ↓
INT8
 ↓
INT4
```

Document for each format:

-   representation;
-   scale;
-   zero point if necessary;
-   packing;
-   dequantization;
-   kernels;
-   memory impact;
-   quality impact;
-   performance impact.

------------------------------------------------------------------------

## 21. Phase 17 --- GPU

GPU must only be considered after obtaining a correct CPU
engine.

Architecture:

``` text
Runtime
   |
   +--- CPU Backend
   |
   +--- GPU Backend
```

Tensor operations must be backend-independent as much as possible.

Example:

``` text
Tensor
  |
  v
Backend
  |
  +-- CPU
  |
  +-- CUDA
  |
  +-- Metal
  |
  +-- Vulkan
```

The first GPU backend must be chosen based on ScriptC's actual
capabilities.

------------------------------------------------------------------------

## 22. Architecture Rules

Systematically respect:

### Separation of Concerns

``` text
Tensor
Math
NN
Transformer
Tokenizer
Model
Runtime
Sampler
Backend
```

must not be mixed.

### No Magic

Important dimensions, layouts, and transformations must be
explicit.

### No Unnecessary Dependencies

Do not integrate llama.cpp, TensorRT, or another engine to implement
the core engine.

External tools may be used for:

-   validation;
-   reference model generation;
-   weight conversion;
-   benchmarks.

------------------------------------------------------------------------

## 23. Recommended Project Structure

``` text
scriptc-llm/
│
├── README.md
├── INSTRUCTIONS.md
├── LICENSE
│
├── src/
│   ├── tensor/
│   ├── math/
│   ├── nn/
│   ├── transformer/
│   ├── tokenizer/
│   ├── model/
│   ├── runtime/
│   ├── sampling/
│   └── backend/
│
├── tests/
│   ├── tensor/
│   ├── math/
│   ├── nn/
│   ├── transformer/
│   └── runtime/
│
├── examples/
│   ├── tensor.sc
│   ├── transformer.sc
│   └── generate.sc
│
├── benchmarks/
│
└── tools/
    └── reference/
```

------------------------------------------------------------------------

## 24. Test Rules

No new optimization must be merged without tests.

Every important primitive must have:

``` text
unit test
numerical test
edge-case test
benchmark
```

For complex components:

``` text
ScriptC
   |
   +---- compare ----> PyTorch/reference
```

Numerical differences must be measured, not simply
observed.

------------------------------------------------------------------------

## 25. Milestones

### M0 --- Runtime

-   [ ] Understand ScriptC
-   [ ] Memory allocation
-   [ ] Structures
-   [ ] Files
-   [ ] Tests

### M1 --- Tensor

-   [ ] Tensor
-   [ ] Shape
-   [ ] Stride
-   [ ] View
-   [ ] Matmul
-   [ ] Softmax

### M2 --- Neural Network

-   [ ] Linear
-   [ ] Embedding
-   [ ] RMSNorm
-   [ ] SwiGLU

### M3 --- Transformer

-   [ ] Attention
-   [ ] Causal mask
-   [ ] RoPE
-   [ ] Transformer block

### M4 --- Inference

-   [ ] KV cache
-   [ ] Tokenizer
-   [ ] Model loader
-   [ ] Logits
-   [ ] Sampling
-   [ ] Text generation

### M5 --- Validation

-   [ ] PyTorch reference
-   [ ] Numerical comparison
-   [ ] Regression tests

### M6 --- Performance

-   [ ] Profiling
-   [ ] SIMD
-   [ ] Multithreading
-   [ ] Memory optimization

### M7 --- Compression

-   [ ] FP16/BF16
-   [ ] INT8
-   [ ] INT4

### M8 --- GPU

-   [ ] Backend abstraction
-   [ ] GPU tensors
-   [ ] GPU matmul
-   [ ] GPU attention
-   [ ] End-to-end generation

------------------------------------------------------------------------

## 26. First Concrete Objective

Do not start with Llama.

The first deliverable must be:

``` text
ScriptC
   ↓
Tensor FP32
   ↓
Linear
   ↓
RMSNorm
   ↓
Attention
   ↓
MLP
   ↓
Miniature Transformer
   ↓
logits
```

Then:

``` text
Miniature Transformer
   ↓
tokenizer
   ↓
sampling
   ↓
generate("Hello")
```

Once this chain works, move to a real Llama-like architecture.

------------------------------------------------------------------------

## 27. Fundamental Rule

**Correctness first, performance second.**

The engine must always keep a simple implementation serving as a
reference.

Optimization must progressively replace this implementation without
changing its observable behavior.

The project must be built to be able to precisely answer the question:

> "For this token, why did the engine produce this logit?"

If this question cannot be answered, the engine is too abstract
or insufficiently testable.

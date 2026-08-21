# 🦙 llama.scriptc

> **Moteur d'inférence LLM (Transformer) développé de zéro en TypeScript et compilé nativement avec [ScriptC](https://scriptc.dev/).**

---

## 📖 Sommaire

- [Vue d'ensemble](#-vue-densemble)
- [Architecture du projet](#-architecture-du-projet)
- [Comparatif : `llama.scriptc` vs `llama.cpp`](#-comparatif--llamascriptc-vs-llamacpp)
- [Benchmark Réel de Vitesse](#-benchmark-réel--llamascriptc-vs-llamacpp)
- [Fonctionnalités supportées](#-fonctionnalités-supportées)
- [Installation & Prérequis](#-installation--prérequis)
- [Démarrage rapide](#-démarrage-rapide)
  - [1. Exécuter un modèle GGUF réel (Qwen2.5)](#1-exécuter-un-modèle-gguf-réel-qwen25)
  - [2. Démo autonome](#2-démo-autonome)
  - [3. Lancer la suite de tests](#3-lancer-la-suite-de-tests)
- [Structure du code](#-structure-du-code)
- [Validation & Tests](#-validation--tests)

---

## 🌟 Vue d'ensemble

`llama.scriptc` est une implémentation complète et autonome d'un moteur d'inférence pour modèles de langage de type Transformer (Llama, Qwen2.5, etc.), écrit **entièrement depuis zéro en TypeScript strict** et compilé directement en **code machine natif** via le compilateur [ScriptC](https://scriptc.dev/) de Vercel.

Le projet a été conçu selon le principe directeur : **« Correctness first, performance second »** (priorité à la clarté, la modularité et la correction mathématique exacte).

```text
                                  Fichier GGUF
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
 │ (Qwen/BPE)  │            │  (Embedding Layer)  │ ◄── KV Cache Autoregressif
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
                            │  Texte en streaming │
                            └─────────────────────┘
```

---

## ⚡ Benchmark Réel : `llama.scriptc` vs `llama.cpp`

### Modèle GGUF Qwen2.5-0.5B (24 couches, 500M params)

Mesures sur **Apple Silicon M4** avec `models/qwen2.5-0.5b-instruct-q8_0.gguf` (prompt : `"Hello, my name is"`) :

| Moteur | Backend & Accélération | Vitesse Prefill | Vitesse Decode | Latence/token |
|:---|:---|:---:|:---:|:---:|
| **`llama.cpp` (Metal GPU)** | GPU Metal + Accelerate BLAS | **946.8 tok/s** | **138.6 tok/s** | **7.2 ms** |
| **`llama.cpp` (CPU)** | CPU Multi-thread (ARM NEON SIMD) | **950.1 tok/s** | **81.1 tok/s** | **12.3 ms** |
| **`llama.scriptc` (Notre moteur)** | CPU Scalaire natif (ScriptC / Mono-thread) | **~5.0 tok/s** | **~0.2 – 2.0 tok/s** | **~2 000 ms** |

> ⚠️ **Note** : Le modèle GGUF reste lent car ScriptC compile en code scalaire mono-thread sans SIMD/GPU. Voir le modèle miniature ci-dessous pour les performances optimisées.

### Modèle Miniature (2 couches, 128 dim) — **Version Optimisée**

Mesures sur le même matériel avec `examples/generate.ts` (prompt : `"Bonjour, je m'appelle"`, 20 tokens générés) :

| Version | Backend | Vitesse Génération | Latence/token |
|:---|:---|:---:|:---:|
| **`llama.scriptc` (Optimisé)** | CPU Scalaire (ScriptC) | **~196 tok/s** | **~5.1 ms** |

### 🔍 Pourquoi une telle différence de vitesse sur le modèle GGUF ?

1. **SIMD & Vectorisation (ARM NEON)** : `llama.cpp` utilise des registres vectoriels traitant 4–8 float32/cycle. `llama.scriptc` exécute des boucles scalaires (1 float/cycle).
2. **Accélération GPU (Metal)** : `llama.cpp` délègue les grosses matmuls (ex: `lm_head` 151k) au GPU Apple Silicon.
3. **Multi-threading (OpenMP/pthreads)** : `llama.cpp` parallélise sur tous les cœurs ; `llama.scriptc` est mono-thread.
4. **Vocation pédagogique** : `llama.scriptc` privilégie la **transparence mathématique** (chaque softmax, RoPE, projection Q/K/V écrit en TypeScript pur).

### 🛠️ Comment reproduire les benchmarks :
```bash
# 1. Modèle GGUF Qwen2.5 (lent - limité par ScriptC scalaire)
npx scriptc run examples/run_gguf.ts --dynamic

# 2. Modèle miniature (rapide - montre les optimisations)
npx scriptc run examples/generate.ts --dynamic "Bonjour, je m'appelle"

# 3. Tester llama.cpp
llama-cli -m models/qwen2.5-0.5b-instruct-q8_0.gguf -p "Hello, my name is" -n 10 --temp 0.0
```

---

## ⚖️ Comparatif : `llama.scriptc` vs `llama.cpp`

| Critère | `llama.scriptc` (Ce projet) | `llama.cpp` |
|:---|:---|:---|
| **Langage source** | **TypeScript strict** | **C / C++ (C11 / C++14)** |
| **Moteur d'exécution** | Compilateur AOT **ScriptC** (LLVM/clang natif) | Compilé nativement (GCC / Clang / MSVC) |
| **Lisibilité du code** | **Très élevée** (~2 500 lignes modulaires et commentées) | **Complexe** (~100 000+ lignes de code C bas niveau et macros) |
| **Objectif principal** | **Pédagogique, vérifiabilité & architecture propre** | **Performance brute et déploiement industriel** |
| **Support de formats** | **GGUF v3**, format JSON maison | **GGUF v1, v2, v3**, formats historiques GGML |
| **Quantization** | **Q8_0** (déquantification $O(1)$), **F32**, **F16** | **20+ formats** (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, IQ, etc.) |
| **Gestion mémoire** | Tableaux denses typés, structures TypeScript, **Pool de tenseurs** | Gestion manuelle `malloc`/`free`, `mmap`, pool GGML |
| **Accélération matérielle** | CPU scalaire natif (mono-thread) | **SIMD** (AVX2, AVX-512, ARM NEON) + **GPU** (Metal, CUDA, Vulkan, ROCm) |
| **Multi-threading** | Mono-thread | Multi-threadé (OpenMP / pthreads natifs) |
| **Vitesse d'inférence (modèle miniature)** | **~196 tok/s** (optimisé) | ~30 à 150+ tok/s sur CPU/GPU |
| **Vitesse d'inférence (GGUF 500M)** | ~0.5–4 tok/s (limité par ScriptC scalaire) | ~30 à 150+ tok/s sur CPU/GPU |
| **Suite de tests** | **16 suites, 68 tests unitaires (< 30ms)** | Tests de régression CI complets |
| **Optimisations clés** | **Fusion d'opérations, vues zero-copy, buffers réutilisés, boucles optimisées** | SIMD, GPU, multi-thread, quantization avancée |
| **Cas d'usage** | Comprendre chaque équation d'un LLM, prototypage, écosystème TS | Production, serveurs d'inférence, applications mobiles/edge |

---

## ✨ Fonctionnalités supportées

- [x] **Système de Tenseurs N-D** : Strides row-major, gestion de vues (`view()`), slicing, transpositions 2D, broadcasting automatique, **Pool de tenseurs pour réutilisation de buffers**.
- [x] **Math & Normalisation** : Matmul 2D/3D batched **optimisé (boucles i-j-k, accès contigus)**, Softmax numériquement stable **in-place** avec compensation de $\max(x)$.
- [x] **Primitives Réseau** : `Linear` (**biais fusionné, buffer réutilisé**), `Embedding`, `RMSNorm` (**fusion normalisation+scale, buffer réutilisé**), activations `SiLU` et bloc `SwiGLU` (**SiLU×Up fusionné**).
- [x] **Attention Causale** : Masquage triangulaire supérieur ($-10^9$), Multi-Head Attention (MHA) et Grouped-Query Attention (GQA) **avec RoPE in-place, KV Cache zero-copy, softmax fusionné**.
- [x] **Rotary Positional Embeddings (RoPE)** : Fréquences $\theta_i = \text{base}^{-2i/d}$, rotation 2D par paires préservant la norme $L_2$, compatible RoPE base $1\,000\,000$ (Qwen2), **application in-place sans allocation**.
- [x] **KV Cache** : Cache Key-Value avec mise à jour $O(1)$ pour le décodage autoregressif pas-à-pas (gain de complexité de $O(N^2)$ à $O(N)$), **vues au lieu de copies, reset in-place**.
- [x] **Support GGUF v3 & Q8_0** : Lecture binaire, extraction de métadonnées, et déquantification des blocs $32 \times \text{int8} + \text{float16 scale}$.
- [x] **Tokenizer BPE Qwen** : Décodage du vocabulaire de 151 936 tokens et déséchappement des octets multi-octets UTF-8.
- [x] **Stratégies de Sampling** : Greedy ($\text{temp}=0$), Temperature, Top-K ($O(N \cdot K)$ optimisé), Top-P (Nucleus), et Repetition Penalty.

---

## 🚀 Installation & Prérequis

### Prérequis
- **ScriptC** installé globalement :
  ```bash
  npm install -g scriptc
  ```
  *(Ou vérifiez que `scriptc` est dans votre `$PATH$)*

---

## ⚡ Démarrage rapide

### 1. Exécuter un modèle GGUF réel (Qwen2.5)

Pour charger et exécuter le modèle quantifié `models/qwen2.5-0.5b-instruct-q8_0.gguf` :

```bash
# Lancer la génération avec le modèle GGUF réel
npx scriptc run examples/run_gguf.ts --dynamic
```

Exemple de sortie :
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
您好，我是来自中国的一位，我...
----------------------------------------------
```

> ⚠️ **Note** : Le modèle GGUF (24 couches, 500M params) reste lent (~0.5–4 tok/s) car limité par l'exécution scalaire mono-thread de ScriptC. Voir le benchmark ci-dessus pour la comparaison détaillée.

---

### 2. Démo autonome avec modèle miniature

Pour tester le pipeline complet avec un modèle miniature et génération instantanée :

```bash
npx scriptc run examples/generate.ts --dynamic "Bonjour, je m'appelle"
```

Exemple de sortie (version optimisée) :
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

[4/4] Starting generation for prompt: "Bonjour, je m'appelle"
---------------------------------------------------------
Bonjour, je m'appelle(%%?+5!,A7-:/8%C*A29
---------------------------------------------------------

✨ Generation finished in 102ms
   - Prompt tokens    : 5
   - Generated tokens : 20
   - Total tokens     : 25
   - Speed            : 196.1 tokens/sec (5.10 ms/token)

=========================================================
```

---

### 3. Lancer la suite de tests

Le projet inclut 16 suites de tests validant la totalité des opérations mathématiques et neuronales :

```bash
npx scriptc run tests/run_all.ts --dynamic
```

Résultat d'exécution (version optimisée) :
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

## 📂 Structure du code

```text
llama.scriptc/
├── src/
│   ├── tensor/             # Tenseurs N-D, types, strides et vues
│   │   ├── tensor.ts
│   │   ├── shape.ts
│   │   └── dtype.ts
│   ├── math/               # Opérations mathématiques
│   │   ├── elementwise.ts  # add, sub, mul, div, exp, sqrt, rsqrt
│   │   ├── reduction.ts    # sum, mean, max, min, argmax
│   │   ├── matmul.ts       # Matmul 2D / 3D batched
│   │   └── softmax.ts      # Softmax stable
│   ├── nn/                 # Primitives neuronales
│   │   ├── embedding.ts    # Table de lookup
│   │   ├── linear.ts       # Projection dense y = xW^T + b
│   │   ├── rmsnorm.ts      # Normalisation RMS
│   │   ├── activations.ts  # SiLU / Sigmoid
│   │   └── swiglu.ts       # SwiGLU MLP Llama-style
│   ├── transformer/        # Architecture Transformer
│   │   ├── rope.ts         # Rotary Positional Embeddings
│   │   ├── attention.ts    # Multi-Head & Grouped-Query Attention
│   │   ├── block.ts        # Bloc Transformer (Pre-RMSNorm + Residuals)
│   │   └── kv_cache.ts     # KV Cache autoregressif O(1)
│   ├── tokenizer/          # Tokenization
│   │   ├── tokenizer.ts    # Tokenizer standard
│   │   └── gguf_tokenizer.ts # Tokenizer GGUF BPE avec décodage UTF-8
│   ├── model/              # Modèle & Chargement
│   │   ├── config.ts       # ModelConfig
│   │   ├── transformer.ts  # Modèle Transformer complet
│   │   ├── gguf.ts         # Parser GGUF v3 & déquantification Q8_0
│   │   └── loader.ts       # Sauvegarde/chargement JSON & GGUF
│   ├── sampling/           # Sampler
│   │   └── sampler.ts      # Greedy, Temperature, Top-K, Top-P, Rep Penalty
│   └── runtime/            # Pipeline d'inférence
│       └── generator.ts    # Générateur de texte complet (Prefill + Decode)
├── tests/                  # 16 suites de tests automatisés
│   ├── run_all.ts          # Lanceur global
│   ├── tensor/
│   ├── nn/
│   ├── transformer/
│   ├── tokenizer/
│   ├── model/
│   └── sampling/
├── examples/               # Exemples exécutables
│   ├── run_gguf.ts         # Démo Qwen2.5 GGUF réelle
│   └── generate.ts         # Démo avec modèle miniature
└── models/                 # Modèles GGUF (ex: qwen2.5-0.5b-instruct-q8_0.gguf)
```

---

## 🔬 Validation & Tests

1. **Équivalence mathématique du KV Cache** : Vérification formelle que le décodage pas-à-pas avec KV Cache produit un résultat **strictement identique ($< 10^{-6}$)** au recalcul complet sans cache.
2. **Causalité stricte de l'attention** : Vérification qu'aucune fuite d'information future n'altère le passé (masquage causal triangulaire).
3. **Référence PyTorch** : Un script de référence [`tools/reference/pytorch_reference.py`](tools/reference/pytorch_reference.py) permet d'exporter les poids et activations couche par couche pour une comparaison numérique directe.

---

## 📜 Licence

Projet open-source sous licence MIT.

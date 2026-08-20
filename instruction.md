# Instructions --- Construire un moteur d'inférence LLM en ScriptC

## 1. Objectif

Construire **depuis zéro** un moteur d'inférence pour modèles de langage
de type Transformer en ScriptC.

Le projet doit privilégier :

-   la compréhension complète du fonctionnement d'un LLM ;
-   une architecture simple et modulaire ;
-   la correction numérique avant la performance ;
-   des tests unitaires et des tests de référence ;
-   une optimisation progressive ;
-   l'absence de dépendance à un moteur d'inférence existant pour le
    cœur du runtime.

Le premier objectif n'est pas de battre llama.cpp en performance. Le
premier objectif est d'obtenir une implémentation **correcte, lisible et
vérifiable** capable de générer du texte.

------------------------------------------------------------------------

## 2. Principe de développement

Toujours avancer dans cet ordre :

1.  Comprendre le composant.
2.  Définir ses structures de données.
3.  Implémenter une version simple.
4.  Écrire les tests.
5.  Comparer les résultats avec une référence.
6.  Mesurer les performances.
7.  Optimiser uniquement après validation.

Ne jamais optimiser une opération dont la correction n'a pas été
démontrée.

------------------------------------------------------------------------

## 3. Architecture cible

Le moteur doit progressivement évoluer vers cette architecture :

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

## 4. Phase 0 --- Comprendre ScriptC

Avant d'implémenter le moteur LLM, documenter précisément les capacités
de ScriptC.

À déterminer :

-   système de types ;
-   pointeurs ;
-   références ;
-   allocation mémoire ;
-   tableaux ;
-   structures ;
-   modules ;
-   fonctions ;
-   gestion des erreurs ;
-   fichiers ;
-   I/O ;
-   conversions numériques ;
-   `float32` ;
-   `float16` / `bfloat16` si disponibles ;
-   threads ;
-   SIMD ;
-   accès natif ;
-   FFI ;
-   GPU éventuel ;
-   système de build ;
-   système de tests ;
-   profiling/debugging.

Créer un petit programme de test pour chaque capacité nécessaire.

**Ne pas supposer qu'une fonctionnalité existe. La vérifier.**

------------------------------------------------------------------------

## 5. Phase 1 --- Système Tensor

Créer un type `Tensor`.

Minimum recommandé :

``` text
Tensor
├── data
├── dtype
├── shape
├── stride
└── size
```

Le tensor doit supporter au minimum :

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

Puis implémenter :

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

Commencer exclusivement avec `float32`.

### Contraintes

-   éviter les copies inutiles ;
-   distinguer clairement ownership et views ;
-   vérifier les dimensions ;
-   gérer correctement les strides ;
-   tester les cas limites.

------------------------------------------------------------------------

## 6. Phase 2 --- Tests numériques

Chaque opération Tensor doit disposer de tests.

Exemple :

``` text
A = [[1, 2],
     [3, 4]]

B = [[5, 6],
     [7, 8]]

C = A @ B
```

Résultat attendu :

``` text
[[19, 22],
 [43, 50]]
```

Les tests doivent également couvrir :

-   tenseurs 1D ;
-   tenseurs 2D ;
-   tenseurs 3D ;
-   dimensions incompatibles ;
-   tenseurs vides si le runtime les autorise ;
-   valeurs négatives ;
-   grandes valeurs ;
-   petites valeurs ;
-   NaN/Inf lorsque pertinent.

------------------------------------------------------------------------

## 7. Phase 3 --- Neural Network Primitives

Implémenter :

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

Utiliser une implémentation numériquement stable :

``` text
softmax(x_i) =
    exp(x_i - max(x))
    ------------------
    sum(exp(x_j - max(x)))
```

### SwiGLU

Implémenter la variante utilisée par les architectures Llama-like.

------------------------------------------------------------------------

## 8. Phase 4 --- Attention

Implémenter d'abord une attention causale simple.

``` text
Q = XWq
K = XWk
V = XWv

Attention(Q,K,V) =
    softmax(QKᵀ / sqrt(d))V
```

Ajouter ensuite :

-   causal mask ;
-   multi-head attention ;
-   RoPE ;
-   grouped-query attention si nécessaire.

Avant toute optimisation, vérifier les résultats sur de très petits
tenseurs.

------------------------------------------------------------------------

## 9. Phase 5 --- RoPE

Implémenter Rotary Positional Embeddings.

Vérifier :

-   fréquence ;
-   position ;
-   dimension ;
-   paire de dimensions ;
-   comportement à la position 0 ;
-   cohérence avec une implémentation de référence.

Créer des tests numériques indépendants.

------------------------------------------------------------------------

## 10. Phase 6 --- Transformer Block

Construire :

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

Le Transformer doit être composé de modules indépendants.

Éviter de mettre toute la logique dans une seule fonction.

------------------------------------------------------------------------

## 11. Phase 7 --- KV Cache

Implémenter le KV cache avant de viser une génération efficace.

Objectif :

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

Le cache doit éviter de recalculer les clés et valeurs historiques à
chaque token.

Documenter précisément :

-   layout mémoire ;
-   dimensions ;
-   capacité ;
-   position courante ;
-   gestion des contextes longs ;
-   allocation ;
-   réutilisation.

------------------------------------------------------------------------

## 12. Phase 8 --- Tokenizer

Implémenter ou intégrer un tokenizer uniquement après avoir défini
clairement le format nécessaire.

Le tokenizer doit fournir :

``` text
encode(text) -> token_ids

decode(token_ids) -> text
```

Commencer avec un format simple.

Ensuite supporter les tokenizers nécessaires aux modèles ciblés.

------------------------------------------------------------------------

## 13. Phase 9 --- Model Loader

Créer une abstraction :

``` text
ModelLoader
├── metadata
├── tensors
└── model configuration
```

Le loader doit récupérer :

-   vocab size ;
-   hidden size ;
-   nombre de couches ;
-   nombre de heads ;
-   nombre de KV heads ;
-   head dimension ;
-   contexte ;
-   paramètres RoPE ;
-   dtype ;
-   poids.

Commencer avec **un format de fichier simple et contrôlé par le
projet**.

Ajouter GGUF ou d'autres formats ensuite.

Ne pas commencer par supporter tous les formats existants.

------------------------------------------------------------------------

## 14. Phase 10 --- Premier modèle

Le premier modèle cible doit être volontairement petit.

Exemple :

``` text
vocab       = 1000
hidden      = 128
layers      = 2
heads       = 4
kv_heads    = 4
intermediate = 512
```

L'objectif est de pouvoir faire :

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
texte
```

------------------------------------------------------------------------

## 15. Phase 11 --- Validation avec PyTorch

Créer un petit modèle de référence en Python/PyTorch.

Pour chaque étape comparer :

``` text
ScriptC output
vs
PyTorch output
```

Comparer notamment :

-   embeddings ;
-   RMSNorm ;
-   Linear ;
-   RoPE ;
-   Q/K/V ;
-   attention ;
-   MLP ;
-   Transformer block ;
-   logits.

Utiliser une tolérance numérique explicite.

Exemple :

``` text
abs(a - b) < epsilon
```

et, lorsque nécessaire :

``` text
relative_error(a, b) < epsilon
```

------------------------------------------------------------------------

## 16. Phase 12 --- Sampling

Implémenter progressivement :

``` text
greedy
temperature
top-k
top-p
repetition penalty
```

Le sampler doit être indépendant du Transformer.

Interface conceptuelle :

``` text
token = sampler.sample(logits)
```

------------------------------------------------------------------------

## 17. Phase 13 --- Génération

Créer une API minimale :

``` text
generate(
    prompt,
    max_tokens,
    temperature,
    top_k,
    top_p
)
```

Exemple d'utilisation :

``` text
generate(
    "Bonjour, je m'appelle",
    100,
    0.7,
    40,
    0.9
)
```

------------------------------------------------------------------------

## 18. Phase 14 --- Performance

Ne commencer l'optimisation qu'après validation complète.

Mesurer séparément :

``` text
Model loading
Prefill
Decode
Tokens/sec
Latency/token
Memory usage
Peak memory
```

Identifier les hotspots avec un profiler.

Les premières optimisations doivent cibler :

1.  `matmul`
2.  attention
3.  copies mémoire
4.  KV cache
5.  allocation
6.  opérations élémentaires

------------------------------------------------------------------------

## 19. Phase 15 --- SIMD

Une fois la version naïve correcte :

``` text
Scalar
  ↓
SIMD
  ↓
Multithreading
  ↓
Optimized kernels
```

Déterminer les possibilités offertes par ScriptC :

``` text
AVX2
AVX-512
NEON
SVE
```

selon la plateforme.

Ne jamais casser l'implémentation de référence lors d'une optimisation.

Conserver une version simple comme oracle de test.

------------------------------------------------------------------------

## 20. Phase 16 --- Quantification

Ajouter progressivement :

``` text
FP32
 ↓
FP16/BF16
 ↓
INT8
 ↓
INT4
```

Documenter pour chaque format :

-   représentation ;
-   échelle ;
-   zero point si nécessaire ;
-   packing ;
-   déquantification ;
-   kernels ;
-   impact mémoire ;
-   impact qualité ;
-   impact performance.

------------------------------------------------------------------------

## 21. Phase 17 --- GPU

Le GPU ne doit être envisagé qu'après avoir obtenu un moteur CPU
correct.

Architecture :

``` text
Runtime
   |
   +--- CPU Backend
   |
   +--- GPU Backend
```

Les opérations Tensor doivent être indépendantes du backend autant que
possible.

Exemple :

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

Le premier backend GPU doit être choisi selon les capacités réelles de
ScriptC.

------------------------------------------------------------------------

## 22. Règles d'architecture

Respecter systématiquement :

### Séparation des responsabilités

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

ne doivent pas être mélangés.

### Pas de magie

Les dimensions, layouts et transformations importantes doivent être
explicites.

### Pas de dépendance inutile

Ne pas intégrer llama.cpp, TensorRT ou un autre moteur pour implémenter
le cœur du moteur.

Des outils externes peuvent être utilisés pour :

-   validation ;
-   génération de modèles de référence ;
-   conversion de poids ;
-   benchmarks.

------------------------------------------------------------------------

## 23. Structure de projet recommandée

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

## 24. Règles de test

Aucune nouvelle optimisation ne doit être fusionnée sans tests.

Chaque primitive importante doit avoir :

``` text
unit test
numerical test
edge-case test
benchmark
```

Pour les composants complexes :

``` text
ScriptC
   |
   +---- compare ----> PyTorch/reference
```

Les différences numériques doivent être mesurées, pas simplement
observées.

------------------------------------------------------------------------

## 25. Milestones

### M0 --- Runtime

-   [ ] Comprendre ScriptC
-   [ ] Allocation mémoire
-   [ ] Structures
-   [ ] Fichiers
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

## 26. Premier objectif concret

Ne pas commencer par Llama.

Le premier livrable doit être :

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
Transformer miniature
   ↓
logits
```

Puis :

``` text
Transformer miniature
   ↓
tokenizer
   ↓
sampling
   ↓
generate("Hello")
```

Une fois cette chaîne fonctionnelle, passer à une architecture
Llama-like réelle.

------------------------------------------------------------------------

## 27. Règle fondamentale

**Correctness first, performance second.**

Le moteur doit toujours conserver une implémentation simple servant de
référence.

L'optimisation doit remplacer progressivement cette implémentation sans
changer son comportement observable.

Le projet doit être construit de manière à pouvoir répondre précisément
à la question :

> « Pour ce token, pourquoi le moteur a-t-il produit ce logit ? »

Si cette question ne peut pas être répondue, le moteur est trop abstrait
ou insuffisamment testable.

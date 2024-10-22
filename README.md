# TP Similarity

TP Similarity is a Python package designed to compute Transition Probability (TP) similarities between nodes in a network. This package provides various methods to estimate these similarities, including exact computation, estimated methods, and node2vec-based cosine similarity.

## Installation

You can install the package via pip:

```bash
pip install tpsimilarity
```

## Features

- **TP (Exact Transition Probabilities):** Computes the exact transition probabilities between nodes in a graph.
- **Estimated TP:** Provides an estimation of the transition probabilities using random walks.
- **Shortest Paths TP:** Computes transition probabilities specifically along the shortest paths.
- **Node2Vec Similarity:** Computes cosine similarity between node embeddings generated by node2vec.

## Usage

Below is an example of how to use the functions provided by the `tpsimilarity` package. 

### Importing the Package

```python
import tpsimilarity.similarity
```

### Example Usage

#### Compute Exact TP

```python
tpsim = tpsimilarity.similarity.TP(g,
                                sources=sources,
                                targets=targets,
                                walk_length=walk_length)
```

#### Compute Estimated TP

```python
etp = tpsimilarity.similarity.estimatedTP(g,
                                sources=sources,
                                targets=targets,
                                walk_length=walk_length,
                                walks_per_source=1_000_000,
                                batch_size=10_000,
                                return_type="matrix",
                                degreeNormalization=True,
                                progressBar=None)
```

#### Compute Node2Vec Similarity

```python
node2vec_sim = tpsimilarity.similarity.node2vec(g,
                    sources=sources,
                    targets=targets, 
                    dimensions=64,
                    walk_length=walk_length,
                    context_size=10,
                    workers=24,
                    batch_walks=100000,
                    return_type="matrix",
                    progressBar=None)
```

#### Compute Shortest Paths TP

```python
sp_tp = tpsimilarity.similarity.shortestPathsTP(g,
                    sources=sources,
                    targets=targets,
                    walk_length=walk_length)
```

### Parameters

- **graph (igraph.Graph):** The graph on which to compute the similarities.
- **sources (list):** List of source vertices.
- **targets (list):** List of target vertices.
- **walk_length (int):** The length of the random walks.
- **return_type (str):** The type of return value (`list`, `matrix`, `dict`).
- **degreeNormalization (bool):** Whether to normalize by the degree of the target node.
- **precalculated_vectors (np.array):** Precomputed node embeddings for node2vec.
- **dimensions (int):** The number of dimensions for node embeddings in node2vec.
- **context_size (int):** The context size for node2vec.
- **workers (int):** Number of parallel workers for node2vec.
- **train_epochs (int):** Number of training epochs for node2vec.
- **batch_walks (int):** Number of walks per batch for node2vec.
- **progressBar (function, TQDM, bool):** Progress bar or function for tracking progress.

## Authors

- Attila Varga
- Sadamori Kojaku
- Filipi N. Silva

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.



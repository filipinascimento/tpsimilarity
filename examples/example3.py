import networkx as nx
import igraph as ig
from tpsimilarity import similarity

# Create a sample graph using NetworkX
G_nx = nx.karate_club_graph()

# Convert NetworkX graph to iGraph
G = ig.Graph.from_networkx(G_nx)

# Define sources and targets
sources = [0, 1, 2]  # Source nodes
targets = [3, 4, 5]  # Target nodes

# Compute node2vec similarities
node2vec_sim = similarity.node2vec(
    graph=G,
    sources=sources,
    targets=targets,
    dimensions=64,
    walk_length=10,
    context_size=5,
    workers=4,
    batch_walks=100,
    return_type="matrix",
    progressBar=True
)

# Print the results
print("Node2Vec Similarities:")
print(node2vec_sim)

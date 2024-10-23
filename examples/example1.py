import igraph as ig
from tpsimilarity import similarity

# Create a sample graph using iGraph
G = ig.Graph.Famous('Zachary')

# Define sources and targets
sources = [0, 1, 2]  # Source nodes
targets = [3, 4, 5, 6]  # Target nodes

# Compute exact TP similarities
tp_sim = similarity.TP(
    graph=G,
    sources=sources,
    targets=targets,
    window_length=5
)

# Print the results
print("Exact TP Similarities:")
print(tp_sim)

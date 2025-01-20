import igraph as ig
from tpsimilarity import similarity

# Create a sample graph using iGraph
G = ig.Graph.Famous('Zachary')

# Define sources and targets
sources = [10, 11, 12]  # Source nodes
targets = [10, 11, 12]  # Target nodes

# Compute exact TP similarities
estimated_tp = similarity.estimatedTP(
    graph=G,
    sources=sources,
    targets=targets,
    window_length=5,
    walks_per_source=1000,
    batch_size=100,
    return_type="matrix",
    degreeNormalization=True,
    progressBar=True
)


# Print the results
print("Estimated TP Similarities:")
print(estimated_tp)

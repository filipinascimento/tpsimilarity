import igraph as ig
from tpsimilarity import similarity

# Create a sample graph using iGraph
G = ig.Graph.Famous('Zachary')

# Define sources and targets
sources = [0, 1, 2]  # Source nodes
targets = [3, 4, 5, 6]  # Target nodes


spTP = similarity.shortestPathsTP(
            graph=G,
            sources=sources,
            targets=targets,
            window_length=100
        )


# Print the results
print("Shortest path TP Similarities:")
print(spTP)

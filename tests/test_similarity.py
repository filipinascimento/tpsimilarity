import unittest
import igraph as ig
import numpy as np
from tpsimilarity import similarity

class TestSimilarityFunctions(unittest.TestCase):

    def setUp(self):
        # Create a sample graph using iGraph
        self.G = ig.Graph.Famous('Zachary')
        self.sources = [0, 1, 2]
        self.targets = [3, 4, 5, 6]

    def test_TP(self):
        tp_sim = similarity.TP(
            graph=self.G,
            sources=self.sources,
            targets=self.targets,
            window_length=5
        )
        self.assertEqual(tp_sim.shape, (3, 4))
        self.assertTrue(np.all(tp_sim >= 0))

    def test_estimatedTP(self):
        estimated_tp = similarity.estimatedTP(
            graph=self.G,
            sources=self.sources,
            targets=self.targets,
            window_length=5,
            walks_per_source=1000,
            batch_size=100,
            return_type="matrix",
            degreeNormalization=True,
            progressBar=False
        )
        self.assertEqual(estimated_tp.shape, (3, 4))
        self.assertTrue(np.all(estimated_tp >= 0))

    def test_shortestPathsTP(self):
        sp_tp = similarity.shortestPathsTP(
            graph=self.G,
            sources=self.sources,
            targets=self.targets,
            window_length=5
        )
        self.assertEqual(sp_tp.shape, (3, 4))
        self.assertTrue(np.all(sp_tp >= 0))

    def test_node2vec(self):
        node2vec_sim = similarity.node2vec(
            graph=self.G,
            sources=self.sources,
            targets=self.targets,
            dimensions=64,
            window_length=10,
            context_size=5,
            workers=4,
            batch_walks=100,
            return_type="matrix",
            progressBar=False
        )
        self.assertEqual(node2vec_sim.shape, (3, 4))
        self.assertTrue(np.all(node2vec_sim >= 0))

if __name__ == '__main__':
    unittest.main()

import igraph as ig
import numpy as np
from scipy import sparse
from tqdm.auto import tqdm
from collections import Counter
from scipy.spatial import distance as spdistance

import cxrandomwalk as rw
import fastnode2vec

def shortestPathsTP(graph, sources = None, targets=None, window_length = 10, return_type = "matrix", progressBar = None):
    """
    Compute shortest paths TP in a graph.
    It is defined by the average lower bounds for the transition probability of the shortest paths from sources to targets.
    It is computed as the product of the inverse node degrees along the path.

    Parameters
    ----------
    graph : igraph.Graph
        The graph.
    sources : list, optional
        The source vertices. If None, the shortest paths from all vertices are computed.
    targets : list, optional
        The target vertices. If None, the shortest paths to all vertices are computed.
    window_length : int, optional
        The window length.
    return_type : str, optional
        The type of the return. It can be "list", "matrix", "dict".
    progressBar : function, TQDM, bool, optional
        A progress bar. If None, no progress bar is shown.
        If function, it should receive three arguments: current, total, and label.
        If TQDM, it will use the provided TQDM progress bar.
        If bool, it will show a progress bar if True.
    Returns
    -------
    list, dict, matrix
        The shortest paths TP.

    """
    if(sources is None):
        sources = np.arange(0,graph.vcount())
    if(targets is None):
        targets = np.arange(0,graph.vcount())
    
    shortestPaths = [[[] for j in range(0,len(targets))] for i in range(0,len(sources))]
    targets2Index = {targets[i]:i for i in range(0,len(targets))}
    for sourceIndex, sourceNode in enumerate(sources):
        for path in graph.get_all_shortest_paths(sourceNode, to=targets):
            if(len(path)>0 and len(path)<=window_length):
                shortestPaths[sourceIndex][targets2Index[path[-1]]].append(path)
    shortestTPProbabilities = np.zeros((len(sources),len(targets)))
    degrees = np.array(graph.degree())
    for sourceNode in range(0,len(sources)):
        for targetNode in range(0,len(targets)):
            for path in shortestPaths[sourceNode][targetNode]:
                if(path):
                    shortestTPProbabilities[sourceNode,targetNode] += np.prod(1/degrees[path])
    if(return_type == "list"):
        # return a list of [(source,target,shortestTPProbabilities),...]
        return [(sources[i],targets[j],shortestTPProbabilities[i][j]) for i in range(0,len(sources)) for j in range(0,len(targets))]
    if(return_type == "dict"):
        # return a dict of {(source,target):shortestTPProbabilities,...}
        return {(sources[i],targets[j]):shortestTPProbabilities[i][j] for i in range(0,len(sources)) for j in range(0,len(targets))}
    if(return_type == "matrix"):
        # return a matrix of shortestTPProbabilities
        return shortestTPProbabilities


def TP(graph, sources = None, targets=None, window_length=10, return_type = "matrix", progressBar = None):
    """
    Compute exact TP (transition probabilities) in a graph.

    Parameters
    ----------
    graph : igraph.Graph
        The graph.
    sources : list, optional
        The source vertices. If None, the shortest paths from all vertices are computed.
    targets : list, optional
        The target vertices. If None, the shortest paths to all vertices are computed.
    window_length : int, optional
        The window length.
    return_type : str, optional
        The type of the return. It can be "list", "matrix", "dict".
    progressBar : function, TQDM, bool, optional
        A progress bar. If None, no progress bar is shown.
        If function, it should receive three arguments: current, total, and label.
        If TQDM, it will use the provided TQDM progress bar.
        If bool, it will show a progress bar if True.

    Returns
    -------
    list, dict, matrix
        The exact TP.

    """
    if(sources is None):
        sources = np.arange(0,graph.vcount())
    if(targets is None):    
        targets = np.arange(0,graph.vcount())

    A = np.array(graph.get_adjacency().data)
    # by Sadamori
    deg = np.array(A.sum(axis = 1)).reshape(-1)
    P = sparse.diags(1/deg) @ A # transition matrix transposed?
    w = np.ones(window_length)
    w = w / np.sum(w)
    Pt = sparse.csr_matrix(sparse.diags(np.ones(P.shape[0]))) # diag 1
    Ps = sparse.csr_matrix(sparse.diags(np.zeros(P.shape[0]))) # empty
    for i in tqdm(range(window_length)):
        Pt = P @ Pt
        Ps = Ps + w[i] * Pt
        # print(i+1)
    degrees = np.array(graph.degree())
    if(return_type == "list"):
        # return a list of [(source,target,shortestTPProbabilities),...]
        return [(sources[i],targets[j],Ps[i,j]/degrees[j]) for i in range(0,len(sources)) for j in range(0,len(targets))]
    if(return_type == "dict"):
        # return a dict of {(source,target):shortestTPProbabilities,...}
        return {(sources[i],targets[j]):Ps[i,j]/degrees[j] for i in range(0,len(sources)) for j in range(0,len(targets))}
    if(return_type == "matrix"):
        #reduce matrix to sources vs targets
        Ps = Ps[sources][:, targets]
        return Ps / degrees[targets]
    

def estimatedTP(graph, sources = None, targets=None, window_length=20, walks_per_source=1_000_000,
                batch_size=10_000, return_type = "matrix", degreeNormalization=True, progressBar = None):
    """
    Compute estimated TP (transition probabilities) in a graph.
    
    Parameters
    ----------
    graph : igraph.Graph
        The graph.
    sources : list, optional
        The source vertices. If None, the shortest paths from all vertices are computed.
    targets : list, optional
        The target vertices. If None, the shortest paths to all vertices are computed.
    window_length : int, optional
        The window length.
    walks_per_source : int, optional
        The number of walks per source.
    batch_size : int, optional
        The batch size (how many walks are computed at once).
    return_type : str, optional
        The type of the return. It can be "list", "matrix", "dict".
    degreeNormalization : bool, optional
        Normalize by the degree of the target node.
    progressBar : function, TQDM, bool, optional
        A progress bar. If None, no progress bar is shown.
        If function, it should receive three arguments: current, total, and label.
        If TQDM, it will use the provided TQDM progress bar.
        If bool, it will show a progress bar if True.

    Returns
    -------
    list, dict, matrix
        The estimated TP.
    """
    if(sources is None):
        sources = np.arange(0,graph.vcount())
    if(targets is None):    
        targets = np.arange(0,graph.vcount())

    

    vertexCount = graph.vcount()
    edges = graph.get_edgelist()

    agent = rw.Agent(vertexCount,edges,False)

    degrees = np.array(graph.degree())

    hits = agent.walkHits(nodes=list(sources),
                      q=1.0,
                      p=1.0,
                      walksPerNode=walks_per_source,
                      batchSize=batch_size,
                      windowSize=window_length,
                      verbose=False,
                      updateInterval=1000,)
    
    totalHitsPerNode = window_length * walks_per_source

    probabilities = hits / totalHitsPerNode
    if degreeNormalization:
        # divide by degree of each target
        probabilities = probabilities / degrees[np.newaxis, :]

    # np array of shape (sources, vertexCount)

    if(return_type == "list"):
        # return a list of [(source,target,shortestTPProbabilities),...]
        return [(sources[i],targets[j],probabilities[i][j]) for i in range(0,len(sources)) for j in range(0,len(targets))]
    if(return_type == "dict"):
        # return a dict of {(source,target):shortestTPProbabilities,...}
        return {(sources[i],targets[j]):probabilities[i][j] for i in range(0,len(sources)) for j in range(0,len(targets))}
    if(return_type == "matrix"):
        # return a matrix of probabilities subset for targets
        return probabilities[:,targets]


def node2vec(graph, precalculated_vectors=None, sources = None, targets=None, 
            dimensions = 64,
            window_length = 40,
            context_size = 10,
            workers=24,
            batch_walks=100000,
            return_type = "matrix", progressBar = None):
    """
    Compute node2vec cosine similarity in a graph.
    
    Parameters
    ----------
    graph : igraph.Graph
        The graph.
    sources : list, optional
        The source vertices. If None, the shortest paths from all vertices are computed.
    targets : list, optional
        The target vertices. If None, the shortest paths to all vertices are computed.
    dimensions : int, optional
        The number of dimensions of the embedding.
    window_length : int, optional
        The walk length for the node2vec.
    context_size : int, optional
        The context size.
    workers : int, optional
        The number of workers for parallel processing.
    batch_walks : int, optional
        The number of walks per batch.
    return_type : str, optional
        The type of the return. It can be "list", "matrix", "dict", "embedding".
        if "embedding", it returns the embedding, which can be used as precalculated_vectors.
    progressBar : function, TQDM, bool, optional
        A progress bar. If None, no progress bar is shown.
        If function, it should receive three arguments: current, total, and label.
        If TQDM, it will use the provided TQDM progress bar.
        If bool, it will show a progress bar if True.

    Returns
    -------
    list, dict, matrix, np.array
        The node2vec cosine similarity.
        if return_type is "embedding", it returns the embedding instead.

    """
    if(sources is None):
        sources = np.arange(0,graph.vcount())
    if(targets is None):    
        targets = np.arange(0,graph.vcount())



    vertexCount = graph.vcount()
    edges = graph.get_edgelist()

    if(precalculated_vectors is None):
        if(graph is None):
            raise ValueError("Either precalculated_vectors or graph should be provided")
        graph = fastnode2vec.Graph(edges, directed=False, weighted=False)
        # embedding
        n2v = fastnode2vec.Node2Vec(graph, dim=dimensions, walk_length=window_length, window=context_size, p=1, q=1, workers=workers, batch_walks=batch_walks)
        n2v.train(epochs=80)
    
        ivec = np.zeros((vertexCount,dimensions))
        for i in tqdm(range(vertexCount)):
            ivec[i] = n2v.wv[i]
    else:
        ivec = precalculated_vectors


    if(return_type == "embedding"):
        return ivec

    probabilities = np.zeros((len(sources),len(targets)))
    for sourceNode in range(0,len(sources)):
        for targetNode in range(0,len(targets)):
            probabilities[sourceNode][targetNode] = 1.0-spdistance.cosine(ivec[sourceNode], ivec[targetNode])

    if(return_type == "list"):
        # return a list of [(source,target,probabilities),...]
        return [(sources[i],targets[j],probabilities[i][j]) for i in range(0,len(sources)) for j in range(0,len(targets))]
    if(return_type == "dict"):
        # return a dict of {(source,target):probabilities,...}
        return {(sources[i],targets[j]):probabilities[i][j] for i in range(0,len(sources)) for j in range(0,len(targets))}
    if(return_type == "matrix"):
        # return a matrix of probabilities
        return probabilities

import networkx as nx

def get_all_shortest_paths(G):
    """
    Get dictionary of all shortest paths between all pairs of nodes.
    Compatible with future NetworkX versions.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary mapping (source, target) pairs to shortest paths
    """
    # Use dict() to ensure we get a dictionary regardless of NetworkX version
    return dict(nx.shortest_path(G))
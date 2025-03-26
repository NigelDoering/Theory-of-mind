import osmnx as ox
import matplotlib.pyplot as plt
import networkx as nx

# Set OSMnx configuration settings
ox.settings.log_console = True
ox.settings.use_cache = True

# Define the place name for UCSD Campus.
place_name = "University of California, San Diego, La Jolla, CA, USA"

# Download the street network graph for the specified area (directed graph by default).
G = ox.graph_from_place(place_name, network_type="all")

# Convert the directed graph to an undirected graph using NetworkX.
G_undirected = nx.Graph(G)

# Save the directed graph to a GraphML file.
ox.save_graphml(G, filepath="ucsd_campus.graphml")

# Plot the directed street network graph.
fig, ax = ox.plot_graph(G, figsize=(10, 10), node_size=10, edge_color="#444444")

# Optionally, get building footprints for the campus area using the updated function.
gdf = ox.features_from_place(place_name, tags={'building': True})
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, color='lightgrey', edgecolor='black')
plt.title("UCSD Campus Buildings")
plt.show()

# Example downstream task: Compute degree centrality on the undirected graph.
degree_centrality = nx.degree_centrality(G_undirected)
print("Degree Centrality for nodes:")
for node, centrality in degree_centrality.items():
    print(f"Node {node}: {centrality:.4f}")

# Example downstream task: Compute shortest path between two nodes (using edge 'length' as weight).
node_list = list(G_undirected.nodes())
if len(node_list) >= 2:
    source, target = node_list[0], node_list[1]
    try:
        shortest_path = nx.shortest_path(G_undirected, source=source, target=target, weight='length')
        print(f"\nShortest path between {source} and {target}:")
        print(shortest_path)
    except nx.NetworkXNoPath:
        print(f"\nNo path found between {source} and {target}.")

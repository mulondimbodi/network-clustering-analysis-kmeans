import argparse
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn.cluster import KMeans


CLUSTER_COLORS = {
    0: "#e41a1c",
    1: "#377eb8",
    2: "#984ea3",
    3: "#999999",
    4: "#a65628",
    5: "#ff7f00",
    6: "#f781bf",
    7: "#4daf4a",
    8: "#ffff33",
    9: "#a6cee3",
}


def generate_network(seed):
    np.random.seed(seed)

    sizes = [85, 95, 80, 90, 75, 90, 85]
    p_in = [0.30, 0.28, 0.32, 0.29, 0.31, 0.27, 0.30]
    p_out = 0.05

    n_blocks = len(sizes)
    probabilities = np.full((n_blocks, n_blocks), p_out, dtype=float)

    for block_index in range(n_blocks):
        probabilities[block_index, block_index] = p_in[block_index]

    graph = nx.stochastic_block_model(sizes, probabilities, seed=seed)
    base_position = nx.spring_layout(graph, seed=seed, k=0.11, iterations=300)
    nodes = list(graph.nodes())
    coordinates = np.array([base_position[node] for node in nodes])

    coordinates[:, 0] = (coordinates[:, 0] - coordinates[:, 0].min()) / (
        coordinates[:, 0].max() - coordinates[:, 0].min()
    )
    coordinates[:, 1] = (coordinates[:, 1] - coordinates[:, 1].min()) / (
        coordinates[:, 1].max() - coordinates[:, 1].min()
    )

    community_centers = np.array(
        [
            [0.25, 0.45],
            [0.38, 0.68],
            [0.52, 0.74],
            [0.68, 0.55],
            [0.68, 0.28],
            [0.48, 0.20],
            [0.22, 0.22],
        ]
    )

    membership = []
    for block_id, size in enumerate(sizes):
        membership.extend([block_id] * size)
    membership = np.array(membership)

    transformed_coordinates = np.zeros_like(coordinates)
    for block_id in range(len(sizes)):
        index = np.where(membership == block_id)[0]
        local_coordinates = coordinates[index].copy()
        local_coordinates[:, 0] -= local_coordinates[:, 0].mean()
        local_coordinates[:, 1] -= local_coordinates[:, 1].mean()
        local_coordinates *= 0.18
        local_coordinates += np.random.normal(0, 0.025, local_coordinates.shape)
        local_coordinates += community_centers[block_id]
        transformed_coordinates[index] = local_coordinates

    energy = np.random.randint(10, 101, size=len(nodes))

    data = pd.DataFrame(
        {
            "Node": [f"N{node}" for node in nodes],
            "X": transformed_coordinates[:, 0] * 100,
            "Y": transformed_coordinates[:, 1] * 100,
            "Energy": energy,
        }
    )

    return graph, data


def apply_kmeans(data, clusters, seed):
    model = KMeans(n_clusters=clusters, random_state=seed, n_init=10)
    data = data.copy()
    data["Cluster"] = model.fit_predict(data[["X", "Y"]])
    return data, model.cluster_centers_


def build_named_graph(graph):
    named_graph = nx.Graph()

    for source, target in graph.edges():
        named_graph.add_edge(f"N{source}", f"N{target}")

    return named_graph


def save_or_show(path, show):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_clustered_network(graph, data, centroids, output_path, show):
    named_graph = build_named_graph(graph)
    coordinate_map = dict(zip(data["Node"], zip(data["X"], data["Y"])))

    plt.figure(figsize=(12, 12), facecolor="#efefef")
    axis = plt.gca()
    axis.set_facecolor("#efefef")

    nx.draw_networkx_edges(
        named_graph,
        pos=coordinate_map,
        edge_color="black",
        width=0.35,
        alpha=0.10,
    )

    for cluster_id in sorted(data["Cluster"].unique()):
        cluster_nodes = data[data["Cluster"] == cluster_id]["Node"].tolist()
        nx.draw_networkx_nodes(
            named_graph,
            pos=coordinate_map,
            nodelist=cluster_nodes,
            node_color=CLUSTER_COLORS.get(cluster_id % len(CLUSTER_COLORS)),
            node_size=95,
            alpha=0.82,
            linewidths=0.25,
            edgecolors="white",
        )

    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=430,
        c="black",
        marker="X",
        linewidths=1.5,
        label="Centroids",
    )

    plt.title("K-Means Clustered Network with Centroids", fontsize=15)
    plt.axis("off")
    plt.legend()
    save_or_show(output_path, show)


def plot_energy_scatter(data, centroids, output_path, show):
    figure = plt.figure(figsize=(11, 9))
    axis = figure.add_subplot(111, projection="3d")

    centroid_energy = []
    for cluster_id in sorted(data["Cluster"].unique()):
        cluster_data = data[data["Cluster"] == cluster_id]
        centroid_energy.append(cluster_data["Energy"].mean())
        axis.scatter(
            cluster_data["X"],
            cluster_data["Y"],
            cluster_data["Energy"],
            s=12,
            alpha=0.85,
            label=f"Cluster {cluster_id}",
        )

    axis.scatter(
        centroids[:, 0],
        centroids[:, 1],
        centroid_energy,
        s=250,
        c="black",
        marker="X",
        label="Centroids",
    )

    axis.set_title("3D Clustered Network with Node Energy")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Energy")
    axis.legend()
    save_or_show(output_path, show)


def plot_energy_surface(data, centroids, output_path, show):
    figure = plt.figure(figsize=(12, 9))
    axis = figure.add_subplot(111, projection="3d")

    grid_x, grid_y = np.meshgrid(
        np.linspace(data["X"].min(), data["X"].max(), 150),
        np.linspace(data["Y"].min(), data["Y"].max(), 150),
    )
    grid_z = griddata(
        (data["X"], data["Y"]),
        data["Energy"],
        (grid_x, grid_y),
        method="cubic",
    )

    surface = axis.plot_surface(
        grid_x,
        grid_y,
        grid_z,
        cmap=cm.viridis,
        linewidth=0,
        antialiased=True,
        alpha=0.85,
    )

    centroid_energy = [
        data[data["Cluster"] == cluster_id]["Energy"].mean()
        for cluster_id in sorted(data["Cluster"].unique())
    ]

    figure.colorbar(surface, ax=axis, shrink=0.6, aspect=12, label="Energy Level")
    axis.scatter(data["X"], data["Y"], data["Energy"], s=5, alpha=0.4)
    axis.scatter(
        centroids[:, 0],
        centroids[:, 1],
        centroid_energy,
        s=250,
        c="red",
        marker="X",
        label="Centroids",
    )

    axis.set_title("Smooth 3D Energy Surface with K-Means Clusters")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Energy")
    axis.legend()
    save_or_show(output_path, show)


def summarize_clusters(data):
    summary = (
        data.groupby("Cluster")
        .agg(
            node_count=("Node", "count"),
            average_energy=("Energy", "mean"),
            minimum_energy=("Energy", "min"),
            maximum_energy=("Energy", "max"),
        )
        .round(2)
    )

    print("\nCluster Summary")
    print(summary)


def main():
    parser = argparse.ArgumentParser(
        description="Run K-Means clustering analysis on a synthetic dense network."
    )
    parser.add_argument("--clusters", type=int, default=7, help="Number of K-Means clusters.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible results.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated visualizations.")
    parser.add_argument("--show", action="store_true", help="Display plots interactively.")
    parser.add_argument("--no-save", action="store_true", help="Do not save generated plots.")

    args = parser.parse_args()

    graph, data = generate_network(args.seed)
    clustered_data, centroids = apply_kmeans(data, args.clusters, args.seed)
    summarize_clusters(clustered_data)

    output_directory = Path(args.output_dir)
    base_name = f"kmeans_k{args.clusters}"
    save_paths = {
        "network": output_directory / f"{base_name}_network.png",
        "scatter": output_directory / f"{base_name}_energy_scatter.png",
        "surface": output_directory / f"{base_name}_energy_surface.png",
    }

    if args.no_save:
        save_paths = {key: None for key in save_paths}

    plot_clustered_network(graph, clustered_data, centroids, save_paths["network"], args.show)
    plot_energy_scatter(clustered_data, centroids, save_paths["scatter"], args.show)
    plot_energy_surface(clustered_data, centroids, save_paths["surface"], args.show)

    if not args.no_save:
        print(f"\nVisualizations saved to: {output_directory.resolve()}")


if __name__ == "__main__":
    main()

import networkx as nx
import numpy as np
import osmnx as ox
import requests


class CostMatrix:
    """A cost matrix which indexes the cost by node ids"""

    def __init__(self, node_ids=None, matrix=None):
        self.node_ids = list(node_ids) if node_ids is not None else []
        if matrix is not None:
            self._matrix = {
                node_i: {node_j: matrix[i][j] for j, node_j in enumerate(self.node_ids)}
                for i, node_i in enumerate(self.node_ids)
            }
        else:
            self._matrix = {}

    def __getitem__(self, key):
        return self._matrix.get(key, {})

    def update_matrix(self, orig_id, dest_id, cost):
        inner_dict = self._matrix.get(orig_id, None)
        if inner_dict is not None:
            self._matrix[orig_id][dest_id] = cost
            if not self._matrix[orig_id].get(dest_id, None):
                self.node_ids.append(dest_id)
        else:
            self._matrix[orig_id] = {dest_id: cost}
            self.node_ids.extend([orig_id, dest_id])

    def cost(self, orig_id, dest_id):
        return self._matrix[orig_id][dest_id]

    def get(self, orig_id, dest_id, value=None):
        inner_dict = self._matrix.get(orig_id, None)
        if inner_dict is not None:
            return inner_dict.get(dest_id, value)
        else:
            return value

    def sub_matrix(self, sub_node_ids):
        sub_matrix = [[self._matrix[i][j] for j in sub_node_ids] for i in sub_node_ids]
        return CostMatrix(sub_node_ids, sub_matrix)

    def to_nested_list(self):
        return [[self.get(i, j) for j in self.node_ids] for i in self.node_ids]

    def to_flatten_dict(self):
        return {(i, j): self._matrix[i][j] for i in self._matrix for j in self._matrix[i]}


class Network(CostMatrix):
    def __init__(self, controller):
        super().__init__()
        self.owner = controller
        self._graph = nx.MultiDiGraph()
        self.weight_label = "length"

        # wrapping some useful functions from the osmnx package
        self.get_graph_from_polygon = self._save_network_graph(ox.graph_from_polygon)
        self.get_graph_from_address = self._save_network_graph(ox.graph_from_address)
        self.get_graph_from_place = self._save_network_graph(ox.graph_from_place)
        self.get_graph_from_bbox = self._save_network_graph(ox.graph_from_bbox)
        self.load_graph_from_graphml = self._save_network_graph(ox.load_graphml)

        self.save_graph_to_graphml = self._use_network_graph(ox.save_graphml)
        self.save_graph_to_geopackage = self._use_network_graph(ox.save_graph_geopackage)
        self.plot_route_on_graph = self._use_network_graph(ox.plot_graph_route)
        self.plot_graph = self._use_network_graph(ox.plot_graph)
        self.get_nearest_nodes = self._use_network_graph(ox.nearest_nodes)

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, G):
        self._graph = G

    def get_random_nodes(self, n=1, all_reachable=False):
        """randomly pick up a number of nodes
        args:
            n, the number of nodes
        return:
            node id or a list of node ids"""
        from core.utils.randomopers import random as rnd
        if all_reachable:
            i = 1
            nodes_list = list(self._graph.nodes)
            selected_nodes = [rnd.choice(nodes_list)]
            while i < n:
                new_node = rnd.choice(nodes_list)
                if nx.has_path(self._graph, selected_nodes[i - 1], new_node):
                    selected_nodes.append(new_node)
                    i += 1
            return selected_nodes
        else:
            return rnd.choices(list(self._graph.nodes), k=n)

    def get_random_locations(self, n=1, all_reachable=False):
        nodes = self.get_random_nodes(n, all_reachable)
        lon = np.zeros(n, dtype=float)
        lat = np.zeros(n, dtype=float)
        for i, n in enumerate(nodes):
            lon[i] = self.graph.nodes[n]['x']
            lat[i] = self.graph.nodes[n]['y']
        return lon, lat

    def get_location(self, node_id):
        return self._graph.nodes[node_id]["x"], self._graph.nodes[node_id]["y"]

    def get_edge_cost(self, edge_id):
        try:
            cost = self._graph.edges[edge_id][self.weight_label]
        except:
            # The path returned by OSRM may include non-existent edges
            u, v = edge_id[0], edge_id[1]
            cost = self.euclidean_distance_between_nodes(u, v)
            self._graph.add_edge(u, v, length=cost)
        return cost

    def cost(self, orig_id, dest_id, cache=True):
        if self._graph is not None:
            if cache and self.get(orig_id, dest_id):
                cost = self._matrix[orig_id][dest_id]
            else:
                cost = self.shortest_path_cost(orig_id, dest_id)
                self.update_matrix(orig_id, dest_id, cost)
            return cost
        else:
            raise ValueError("No graph initialized in the network")

    def speed(self, orig_id, dest_id):
        """Speed-flow-model"""
        # use constant for testing 30km per hour
        return 30

    def shortest_path_cost(self, orig, dest, method="dijkstra"):
        """compute the cost of the shortest parth between two nodes on the network
        args:
            orig: the id of the origin node
            dest: the id of the destination node
            method: 'dijkstra' or 'bellman-ford'
        return
            path cost: a scalar measured by Network.weight_label
            """
        osrm_url = "http://localhost:{}/route/v1/driving/".format(self.owner.params['osrm_port'])
        lon1, lat1 = self.graph.nodes[orig]['x'], self.graph.nodes[orig]['y']
        lon2, lat2 = self.graph.nodes[dest]['x'], self.graph.nodes[dest]['y']
        bbox = f"{lon1},{lat1};{lon2},{lat2}"
        request_url = f"{osrm_url}{bbox}?steps=true&annotations=nodes"
        headers = {'Connection': 'close'}
        response = requests.get(request_url, headers=headers)
        data = response.json()
        route = data['routes'][0]
        return route['distance']
        # return nx.astar_path_length(self._graph, orig, dest, weight=self.weight_label)
        # return nx.shortest_path_length(self._graph, orig, dest, weight=self.weight_label, method=method)

    def shortest_path(self, orig, dest, method="dijkstra"):
        """find the shortest parth between two nodes on the network
        args:
            orig: the id of the origin node
            dest: the id of the destination node
            method: 'dijkstra' or 'bellman-ford'
        return
            shortest_path: a list of node ids.
            """
        osrm_url = "http://localhost:{}/route/v1/driving/".format(self.owner.params['osrm_port'])
        lon1, lat1 = self.graph.nodes[orig]['x'], self.graph.nodes[orig]['y']
        lon2, lat2 = self.graph.nodes[dest]['x'], self.graph.nodes[dest]['y']
        bbox = f"{lon1},{lat1};{lon2},{lat2}"
        request_url = f"{osrm_url}{bbox}?steps=true&annotations=nodes"
        headers = {'Connection': 'close'}
        response = requests.get(request_url, headers=headers)
        data = response.json()
        path = data['routes'][0]['legs'][0]['annotation']['nodes']
        return path
        # return nx.astar_path(self._graph, orig, dest, weight=self.weight_label)
        # return nx.shortest_path(self._graph, orig, dest, weight=self.weight_label, method=method)

    def euclidean_distance_between_nodes(self, orig, dest):
        point1 = np.array([self._graph.nodes[orig]['x'], self._graph.nodes[orig]['y']])
        point2 = np.array([self._graph.nodes[dest]['x'], self._graph.nodes[dest]['y']])
        return np.linalg.norm(point1 - point2)

    def update_edge_cost(self):
        """update the cost stored in networkx.edges[edge_id]["weight_label"]
        """
        pass

    def _use_network_graph(self, func):
        """a decorator for re-using osmnx/networkx functions but passing Network._graph to parameter-G"""

        def wrapper(*args, **kwargs):
            return func(self._graph, *args, **kwargs)

        return wrapper

    def _save_network_graph(self, func):
        """a decorator for re-using osmnx/networkx functions but saving returned graph to Network._graph"""

        def wrapper(*args, **kwargs):
            self._graph = func(*args, **kwargs)

        return wrapper

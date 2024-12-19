import abc


class GraphVisualizerInterface(abc.ABC):
    @abc.abstractmethod
    def update_graph(self, nodes):
        """
        Update the graph with new nodes.

        Args:
            nodes (list): A list of new nodes to be added to the graph.
        """
        pass

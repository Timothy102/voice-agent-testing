import json
from typing import Any, Dict, List

import graphviz
import networkx as nx


# Update the visualization code to handle the new structure
class DecisionTreeVisualizer:
    def __init__(self):
        self.dot = graphviz.Digraph(comment="Conversation Flow")
        self.dot.attr(rankdir="TB")

    def create_graph(self, nodes: List[Dict[str, Any]]) -> graphviz.Digraph:
        """Create a graphviz visualization of the conversation flow"""

        self.dot.attr("node", shape="box", style="rounded,filled", fontname="Arial")

        # Add nodes
        for node in nodes:
            node_id = str(node["node_id"])
            content = node["content"]
            speaker = node["speaker"]

            if content == "Start":
                self.dot.node(node_id, content, fillcolor="#E8F5E9", shape="circle")
            elif speaker == "agent":
                self.dot.node(node_id, content, fillcolor="#E3F2FD", shape="diamond")
            else:
                self.dot.node(node_id, content, fillcolor="#FFFFFF")

            # Add edges with customer response labels
            for edge in node["edges"]:
                self.dot.edge(
                    node_id,
                    str(edge["target_node_id"]),
                    label=edge["label"],
                    fontname="Arial",
                    fontsize="10",
                )

        return self.dot

    def save_graph(self, filename: str = "decision_tree", format: str = "png"):
        """Save the graph to a file"""
        self.dot.render(filename, format=format, cleanup=True)

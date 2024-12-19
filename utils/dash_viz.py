import logging
import os
import queue
import threading

import dash_cytoscape as cyto
from dash import Dash, dcc, html
from dash.dependencies import Input, Output


class DynamicGraphVisualizer:
    def __init__(
        self,
        port=8050,
        host="127.0.0.1",
        title="Hamming AI - Real-Time Conversation Flow",
    ):
        self.app = Dash(__name__, suppress_callback_exceptions=True)
        self.port = port
        self.host = host
        self.graph_queue = queue.Queue()
        self.current_elements = []

        # Load Cytoscape layouts
        cyto.load_extra_layouts()

        # Define the layout first
        self.app.layout = html.Div(
            [
                html.H1(title),
                dcc.Interval(
                    id="interval-component", interval=2000, n_intervals=0  # 2 seconds
                ),
                cyto.Cytoscape(
                    id="conversation-graph",
                    layout={"name": "dagre"},
                    style={"width": "100%", "height": "800px"},
                    elements=self.current_elements,
                    stylesheet=[
                        {
                            "selector": "node",
                            "style": {
                                "content": "data(label)",
                                "text-wrap": "wrap",
                                "text-max-width": 100,
                                "font-size": "12px",
                                "text-valign": "center",
                                "text-halign": "center",
                                "width": 120,
                                "height": 60,
                            },
                        },
                        {
                            "selector": ".agent",
                            "style": {
                                "background-color": "#E3F2FD",
                                "shape": "diamond",
                            },
                        },
                        {
                            "selector": ".system",
                            "style": {"background-color": "#E8F5E9", "shape": "circle"},
                        },
                        {
                            "selector": "edge",
                            "style": {
                                "content": "data(label)",
                                "curve-style": "bezier",
                                "target-arrow-shape": "triangle",
                                "font-size": "10px",
                                "text-rotation": "autorotate",
                                "text-margin-y": -10,
                            },
                        },
                    ],
                ),
            ]
        )

        # Register callback after layout is defined
        @self.app.callback(
            Output("conversation-graph", "elements"),
            Input("interval-component", "n_intervals"),
        )
        def update_graph(n):
            try:
                latest_nodes = self.graph_queue.get_nowait()
                self.current_elements = self._convert_nodes_to_elements(latest_nodes)
            except queue.Empty:
                pass
            return self.current_elements

        # Start the server thread
        self.dash_thread = threading.Thread(target=self._run_dash_server, daemon=True)
        self.dash_thread.start()

    def _run_dash_server(self):
        """Run the Dash app in a separate thread"""
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)

        dash_logger = logging.getLogger("dash")
        dash_logger.setLevel(logging.ERROR)

        self.app.run_server(
            debug=False,
            host=self.host,
            port=self.port,
            use_reloader=False,
            dev_tools_hot_reload=False,
        )

    def _convert_nodes_to_elements(self, nodes):
        """Convert nodes to Cytoscape elements"""
        if not nodes:
            return []

        elements = []

        # Create nodes
        for node in nodes:
            node_class = "agent" if node["speaker"] == "agent" else "system"
            elements.append(
                {
                    "data": {"id": str(node["node_id"]), "label": str(node["content"])},
                    "classes": node_class,
                }
            )

        # Create edges
        for node in nodes:
            for edge in node.get("edges", []):
                elements.append(
                    {
                        "data": {
                            "source": str(node["node_id"]),
                            "target": str(edge["target_node_id"]),
                            "label": str(edge.get("label", "")),
                        }
                    }
                )

        return elements

    def update_graph(self, nodes):
        """Update the graph with new nodes"""
        if nodes:
            print(f"Updating graph with {len(nodes)} nodes")
            self.graph_queue.put(nodes)

    def start(self):
        """Start method for compatibility"""
        pass

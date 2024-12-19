import logging
import os
import queue
import threading
from typing import Any, Dict, List

import dash_cytoscape as cyto
from dash import Dash, ctx, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from modules.graph_visualizer.interface import GraphVisualizerInterface


class DynamicGraphVisualizer(GraphVisualizerInterface):
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

        # Initialize cytoscape with all available layouts
        cyto.load_extra_layouts()

        # Define the layout with more robust styling
        self.app.layout = html.Div(
            [
                html.H1(
                    title,
                    style={
                        "textAlign": "center",
                        "marginBottom": "20px",
                        "fontFamily": "Arial, sans-serif",
                    },
                ),
                html.Div(
                    [
                        # Add layout controls
                        html.Label("Layout:"),
                        dcc.Dropdown(
                            id="layout-dropdown",
                            options=[
                                {"label": "Dagre (Top-Down)", "value": "dagre"},
                                {"label": "Concentric", "value": "concentric"},
                                {"label": "Circle", "value": "circle"},
                                {"label": "Cola", "value": "cola"},
                            ],
                            value="dagre",
                            style={"width": "200px", "marginBottom": "10px"},
                        ),
                    ],
                    style={"marginBottom": "20px"},
                ),
                dcc.Interval(
                    id="interval-component",
                    interval=1000,  # Update every second
                    n_intervals=0,
                ),
                cyto.Cytoscape(
                    id="conversation-graph",
                    layout={"name": "dagre", "rankDir": "TB"},
                    style={
                        "width": "100%",
                        "height": "700px",
                        "backgroundColor": "#f8f9fa",
                    },
                    elements=self.current_elements,
                    stylesheet=[
                        {
                            "selector": "node",
                            "style": {
                                "content": "data(label)",
                                "text-wrap": "wrap",
                                "text-max-width": 100,
                                "font-size": "14px",
                                "text-valign": "center",
                                "text-halign": "center",
                                "width": 150,
                                "height": 75,
                                "padding": "10px",
                                "border-width": 2,
                                "border-color": "#666",
                            },
                        },
                        {
                            "selector": ".agent",
                            "style": {
                                "background-color": "#E3F2FD",
                                "shape": "diamond",
                                "border-color": "#1976D2",
                            },
                        },
                        {
                            "selector": ".system",
                            "style": {
                                "background-color": "#E8F5E9",
                                "shape": "circle",
                                "border-color": "#388E3C",
                            },
                        },
                        {
                            "selector": "edge",
                            "style": {
                                "content": "data(label)",
                                "curve-style": "bezier",
                                "target-arrow-shape": "triangle",
                                "font-size": "12px",
                                "text-rotation": "autorotate",
                                "text-margin-y": -10,
                                "line-color": "#666",
                                "target-arrow-color": "#666",
                                "width": 2,
                                "text-outline-color": "#fff",
                                "text-outline-width": 2,
                            },
                        },
                    ],
                ),
            ],
            style={"padding": "20px"},
        )

        # Register callbacks
        self.app.callback(
            Output("conversation-graph", "elements"),
            Input("interval-component", "n_intervals"),
            prevent_initial_call=True,
        )(self.update_elements)

        self.app.callback(
            Output("conversation-graph", "layout"),
            Input("layout-dropdown", "value"),
            prevent_initial_call=True,
        )(self.update_layout)

        # Start the server thread
        self.dash_thread = threading.Thread(target=self._run_dash_server, daemon=True)
        self.dash_thread.start()

    def update_elements(self, n_intervals):
        """Callback to update graph elements"""
        try:
            latest_nodes = self.graph_queue.get_nowait()
            self.current_elements = self._convert_nodes_to_elements(latest_nodes)
            return self.current_elements
        except queue.Empty:
            raise PreventUpdate

    def update_layout(self, layout_name):
        """Callback to update graph layout"""
        return {"name": layout_name, "rankDir": "TB"}

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
        )

    def _convert_nodes_to_elements(
        self, nodes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert nodes to Cytoscape elements with improved handling"""
        if not nodes:
            return []

        elements = []

        # Add nodes with improved styling
        for node in nodes:
            node_id = str(node["node_id"])
            node_class = "agent" if node["speaker"] == "agent" else "system"

            elements.append(
                {
                    "data": {
                        "id": node_id,
                        "label": str(node["content"]),
                        "speaker": node["speaker"],
                    },
                    "classes": node_class,
                    "locked": False,  # Allow nodes to be moved
                }
            )

        # Add edges with improved data handling
        for node in nodes:
            source_id = str(node["node_id"])
            for edge in node.get("edges", []):
                target_id = str(edge["target_node_id"])
                edge_id = f"{source_id}-{target_id}"

                elements.append(
                    {
                        "data": {
                            "id": edge_id,
                            "source": source_id,
                            "target": target_id,
                            "label": str(edge.get("label", "")),
                        }
                    }
                )

        return elements

    def update_graph(self, nodes: List[Dict[str, Any]]) -> None:
        """Update the graph with new nodes"""
        if nodes:
            print(f"Updating graph with {len(nodes)} nodes")
            # Clear the queue before putting new nodes to prevent backlog
            while not self.graph_queue.empty():
                try:
                    self.graph_queue.get_nowait()
                except queue.Empty:
                    break
            self.graph_queue.put(nodes)

    def start(self):
        """Start method for compatibility"""
        pass
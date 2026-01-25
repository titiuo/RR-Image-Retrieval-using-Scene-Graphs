from pyvis.network import Network
import webbrowser
import tempfile
import os
import numpy as np


def visualize_graph(graph_path, index=None):
    import json
    # Load the scene graph
    with open(graph_path, 'r') as f:
        graphs = json.load(f)

    # Select the graph to visualize
    if index is None:
        index = np.random.randint(0, len(graphs))
   
    graph = graphs[index]

    net = Network(notebook=False, cdn_resources='remote')

    # Add nodes for objects
    for obj_id in range(len(graph['objects'])):
        obj_name = graph['objects'][obj_id]['names'][0]
        net.add_node(obj_id, label=obj_name, color="rgb(221, 162, 169)")

        for attr_id in range(len(graph['objects'][obj_id]['attributes'])):
            attr_name = graph['objects'][obj_id]['attributes'][attr_id]['attribute']
            attr_node_id = f"{obj_id}_attr_{attr_id}"
            net.add_node(attr_node_id, label=attr_name, color="rgb(169, 221, 162)")
            net.add_edge(obj_id, attr_node_id,arrows='to',color="lightgray")

    # Add edges for relationships
    for rel_id in range(len(graph['relationships'])):
        rel = graph['relationships'][rel_id]
        subj_id = rel['objects'][0]
        obj_id = rel['objects'][1]
        rel_name = rel['relationship']
        rel_node_id = f"rel_{rel_id}"
        net.add_node(rel_node_id, label=rel_name, color="rgb(201, 238, 253)")
        net.add_edge(subj_id, rel_node_id, arrows='to', color="lightgray")
        net.add_edge(rel_node_id, obj_id, arrows='to', color="lightgray")

    # Save and open the visualization in a web browser
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tf:
        net.save_graph(tf.name)
        webbrowser.open('file://' + os.path.realpath(tf.name))
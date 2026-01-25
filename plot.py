import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import numpy as np


def plot_image_with_bboxes(graph_path,images_path, index=None):
    # Load the scene graph
    with open(graph_path, 'r') as f:
        graphs = json.load(f)

    # Select the graph to plot
    if index is None:
        index = np.random.randint(0, len(graphs))
   
    graph = graphs[index]

    # Load and plot the image
    img = plt.imread(images_path + graph['filename'])
    fig, ax = plt.subplots()
    ax.imshow(img)

    object_color = (221/255, 162/255, 169/255)
    for obj in graph['objects']:
        bbox = obj['bbox']  # [x, y, width, height]
        # Create a Rectangle patch
        rect = patches.Rectangle((bbox['x'], bbox['y']), bbox['w'], bbox['h'],
                                 linewidth=2, edgecolor=object_color, facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
        # Add label
        ax.text(bbox['x'], bbox['y'] - 10, obj['names'][0], color=object_color, weight='bold',
                bbox=dict(facecolor='white', alpha=0.1),fontsize=8)

    plt.show()


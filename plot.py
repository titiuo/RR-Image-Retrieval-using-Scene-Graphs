from pyvis.network import Network
import tempfile
import webbrowser
import os
import subprocess
import platform
import matplotlib.pyplot as plt
import matplotlib.patches as patches




def visualize_graph(scene_graph, obj_vocab, attr_vocab, rel_vocab):
    """
    Visualizes the graph using 100% of the screen height.
    """
    # CHANGE HERE: Use height="100vh" to force full viewport height
    net = Network(notebook=False, cdn_resources='remote', height="100vh", width="100%")

    # --- 1. Add Nodes ---
    for obj_id, obj_data in enumerate(scene_graph.objects):
        obj_name = obj_vocab.get_word(obj_data['class_id'])
        net.add_node(obj_id, label=obj_name, color="rgb(221, 162, 169)")

        for attr_id_code in obj_data['attributes']:
            attr_name = attr_vocab.get_word(attr_id_code)
            attr_node_id = f"{obj_id}_attr_{attr_id_code}"
            net.add_node(attr_node_id, label=attr_name, color="rgb(169, 221, 162)")
            net.add_edge(obj_id, attr_node_id, arrows='to', color="lightgray")

    # --- 2. Add Relationships ---
    for rel_id, rel_data in enumerate(scene_graph.relationships):
        subj_id = rel_data['subject_idx']
        target_obj_id = rel_data['object_idx']
        rel_name = rel_vocab.get_word(rel_data['relation_id'])
        
        rel_node_id = f"rel_{rel_id}"
        net.add_node(rel_node_id, label=rel_name, color="rgb(201, 238, 253)")
        net.add_edge(subj_id, rel_node_id, arrows='to', color="lightgray")
        net.add_edge(rel_node_id, target_obj_id, arrows='to', color="lightgray")

    # --- 3. Save File ---
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tf:
        net.save_graph(tf.name)
        file_path = tf.name

    # --- 4. Cross-Platform Opening ---
    current_os = platform.system()
    release_info = platform.release()

    try:
        # WSL
        if current_os == 'Linux' and 'microsoft' in release_info.lower():
            windows_path = subprocess.check_output(['wslpath', '-w', file_path]).strip().decode('utf-8')
            subprocess.run(['explorer.exe', windows_path])
        # Mac
        elif current_os == 'Darwin':
            subprocess.run(['open', file_path])
        # Windows
        elif current_os == 'Windows':
            os.startfile(file_path)
        # Linux
        else:
            webbrowser.open('file://' + os.path.realpath(file_path))

    except Exception as e:
        print(f"Automatic browser open failed: {e}")



def plot_scene_graph(scene_graph, images_path, obj_vocab):
    """
    Plots the image and bounding boxes for a specific SceneGraph object.
    """
    # 1. Construct full image path
    full_image_path = os.path.join(images_path, scene_graph.filename)
    
    # 2. Load and Plot Image
    try:
        img = plt.imread(full_image_path)
    except FileNotFoundError:
        print(f"Error: Could not find image at {full_image_path}")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)

    # 3. Draw Boxes
    object_color = (221/255, 162/255, 169/255)
    
    for obj in scene_graph.objects:
        bbox = obj['bbox']
        
        # Retrieve name from vocabulary using the class_id
        class_name = obj_vocab.get_word(obj['class_id'])

        # Create Rectangle patch (x, y, w, h)
        rect = patches.Rectangle(
            (bbox['x'], bbox['y']), bbox['w'], bbox['h'],
            linewidth=2, 
            edgecolor=object_color, 
            facecolor='none'
        )
        
        ax.add_patch(rect)
        
        # Add Label
        ax.text(
            bbox['x'], bbox['y'] - 10, 
            class_name, 
            color=object_color, 
            weight='bold',
            bbox=dict(facecolor='white', alpha=0.5, pad=2),
            fontsize=9
        )

    plt.axis('off')
    plt.show()
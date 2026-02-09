import subprocess
import os

def get_bboxes_from_cpp(image_path, bin_path="/Users/titouanduhaze/Desktop/MVA/S2/Reproducible_Research/Geodesic-Object-Proposals/examples/example"):
    if not os.path.exists(bin_path):
        print(f"Error: Executable not found at {bin_path}")
        return []
    
    # Extract the directory where the binary lives
    bin_dir = os.path.dirname(bin_path)

    try:
        # We add 'cwd=bin_dir' so the C++ program can find its '../data/sf.dat' model
        result = subprocess.run(
            [bin_path, image_path], 
            capture_output=True, 
            text=True, 
            cwd=bin_dir
        )
        
        # Debug: if you still get 0, uncomment the next line to see the C++ error
        # print("C++ Output:", result.stdout)
        # print("C++ Error:", result.stderr)

        bboxes = []
        for line in result.stdout.splitlines():
            if line.startswith("BOX:"):
                coords = list(map(int, line.split()[1:]))
                bboxes.append(coords)
                
        return bboxes
    except Exception as e:
        print(f"Python Error: {e}")
        return []

""" # Usage
image_url = "/Users/titouanduhaze/Desktop/MVA/S2/Reproducible_Research/sg_dataset/sg_train_images/52576243_56613d4660_b.jpg"
boxes = get_bboxes_from_cpp(image_url)
print(f"Found {boxes} bounding boxes.") """
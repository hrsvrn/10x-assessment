import os
import shutil
from roboflow import Roboflow
from dotenv import load_dotenv

def download_datasets():
    # Load environment variables
    load_dotenv()
    
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("Error: ROBOFLOW_API_KEY not found in environment variables or .env file.")
        print("Please set it: export ROBOFLOW_API_KEY='your_key'")
        return

    rf = Roboflow(api_key=api_key)
    
    # Define project root and data directory
    # Assuming this script is in the project root
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    datasets = [
        {
            "workspace": "hrsvrn",
            "project": "cracks-3ii36-xl6wn",
            "version": 1,
            "rename_to": "cracks"
        },
        {
            "workspace": "hrsvrn",
            "project": "drywall-join-detect-se2uo",
            "version": 1,
            "rename_to": "drywall_join"
        }
    ]

    for ds_info in datasets:
        print(f"\nDownloading {ds_info['project']}...")
        try:
            project = rf.workspace(ds_info["workspace"]).project(ds_info["project"])
            version = project.version(ds_info["version"])
            
            # Download as semantic segmentation masks
            # This format usually provides images and PNG masks
            dataset = version.download("png-mask-semantic")
            
            # Move/Rename to our data structure
            # dataset.location is the path where it downloaded
            source_path = dataset.location
            dest_path = os.path.join(data_dir, ds_info["rename_to"])
            
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)
            
            # Move the downloaded folder to data/name
            shutil.move(source_path, dest_path)
            print(f"Successfully downloaded and moved to {dest_path}")
            
        except Exception as e:
            print(f"Failed to download {ds_info['project']}: {e}")

if __name__ == "__main__":
    download_datasets()

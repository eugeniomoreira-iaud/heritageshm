import time
import subprocess
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class JupytextSyncHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory:
            return
        
        filepath = event.src_path
        
        # Ignore the 'old' and 'heritageshm' directories
        path_parts = os.path.normpath(filepath).split(os.sep)
        if 'old' in path_parts or 'heritageshm' in path_parts:
            return
        
        # Only trigger on .ipynb or .py files
        if filepath.endswith('.ipynb') or filepath.endswith('.py'):
            # Ignore hidden files, checkpoints, and jupytext temporary files
            basename = os.path.basename(filepath)
            if not basename.startswith('.') and "checkpoint" not in basename and "_tmp_" not in basename:
                if not os.path.exists(filepath):
                    return
                print(f"\n[Saved] Detected change in: {basename}")
                
                # Run the Jupytext sync command on the changed file
                try:
                    # Use shell=True if on Windows and jupytext is not in direct PATH, 
                    # but usually list format works fine with conda
                    subprocess.run(["jupytext", "--sync", filepath], check=True)
                    print(f"Sync complete for {basename}. Watching for more changes...")
                except subprocess.CalledProcessError as e:
                    print(f"Error syncing {basename}: {e}")
                except FileNotFoundError:
                    print("Error: 'jupytext' command not found. Make sure you are in the neuralprophet_env conda environment.")

def initialize_pairs():
    print("Initializing missing .py pairs...")
    count = 0
    # Walk through all directories
    for root, dirs, files in os.walk('.'):
        # Skip the 'old', 'heritageshm' directories and hidden directories
        dirs[:] = [d for d in dirs if d not in ['old', 'heritageshm'] and not d.startswith('.')]
        
        for file in files:
            if file.endswith('.ipynb') and "checkpoint" not in file:
                ipynb_path = os.path.join(root, file)
                py_path = ipynb_path.rsplit('.ipynb', 1)[0] + '.py'
                
                if not os.path.exists(py_path):
                    print(f"Creating missing .py pair for {ipynb_path}")
                    try:
                        subprocess.run(["jupytext", "--set-formats", "ipynb,py:percent", ipynb_path], check=True)
                        count += 1
                    except subprocess.CalledProcessError as e:
                        print(f"Error pairing {ipynb_path}: {e}")
                    except FileNotFoundError:
                        print("Error: 'jupytext' command not found.")
                        return
    print(f"Initialized {count} missing pairs.")

if __name__ == "__main__":
    # First, initialize any missing .py pairs
    initialize_pairs()

    observer = Observer()
    # Watch the current directory recursively
    observer.schedule(JupytextSyncHandler(), path='.', recursive=True)
    observer.start()
    
    print("\nWatching for file saves recursively... (Press Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nWatcher stopped.")
    observer.join()

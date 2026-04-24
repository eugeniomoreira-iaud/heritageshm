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

if __name__ == "__main__":
    observer = Observer()
    # Watch the current directory
    observer.schedule(JupytextSyncHandler(), path='.', recursive=False)
    observer.start()
    
    print("Watching for file saves... (Press Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nWatcher stopped.")
    observer.join()

# Auto Watcher Instructions

This project uses a Python script (`auto_watcher.py`) to automatically keep `.ipynb` notebooks and `.py` script files perfectly synchronized. 

The script uses a library called `watchdog` to monitor your project directory for file changes. Whenever you save a notebook or a Python script, the watcher detects it and instantly runs the `jupytext --sync` command in the background to update its paired file.

## How to Start the Watcher in VS Code

Whenever you start a new coding session, follow these steps to get the auto-sync running in the background:

### 1. Open a new terminal in VS Code
Go to **Terminal -> New Terminal** (or use the shortcut `` Ctrl + ` ``).

### 2. Activate the Conda Environment
It is **crucial** to activate the correct environment first. This ensures that the terminal can find both the `watchdog` package and the `jupytext` command.

Copy and paste the following commands into your terminal and hit Enter:

```powershell
& "C:\ProgramData\anaconda3\shell\condabin\conda-hook.ps1"
conda activate neuralprophet_env
```
*(You should see `(neuralprophet_env)` appear on the left side of your terminal prompt once successful).*

### 3. Run the Watcher Script
Once the environment is active, start the watcher by running:

```powershell
python auto_watcher.py
```

You should see a message saying `Watching for file saves... (Press Ctrl+C to stop)`. 

### You're all set!
You can now leave this terminal running in the background while you work. Every time you hit "Save" on a `.ipynb` or `.py` file, you'll see a quick message pop up in this terminal confirming that it has synced the other file.

**To stop the watcher:** Click on the terminal where it's running and press `Ctrl+C`, or simply click the trash can icon to close the terminal pane.

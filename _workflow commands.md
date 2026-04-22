## Daily Workflow Commands

### 1. Start every session (VS Code terminal)

```powershell
& "C:\ProgramData\anaconda3\shell\condabin\conda-hook.ps1"
conda activate neuralprophet_env
```

### 2. Sync after editing a notebook (`.ipynb` → `.py`)

```powershell
jupytext --sync 02_Proxy_Validation_and_Lags.ipynb
```

### 3. Sync after editing a `.py` file (`.py` → `.ipynb`)

```powershell
jupytext --sync 02_Proxy_Validation_and_Lags.py
```

### 4. Sync all paired files at once

```powershell
jupytext --sync *.ipynb
```
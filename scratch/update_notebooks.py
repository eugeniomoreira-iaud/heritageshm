import json
import os

def update_notebook(path, target_snippet, replacement_snippet):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            if target_snippet in source:
                new_source = source.replace(target_snippet, replacement_snippet)
                # Split back into lines keeping \n
                lines = new_source.splitlines(keepends=True)
                cell['source'] = lines
                modified = True
                break
    
    if modified:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Updated {path}")
    else:
        print(f"No changes made to {path} (target snippet not found)")

nb00 = "00_Sensor_Preprocessing.ipynb"
nb00_t = '''from heritageshm.preprocessing import clean_signal, apply_compensation, temp_compensation'''
nb00_r = '''from heritageshm.preprocessing import clean_signal, apply_compensation, temp_compensation\nfrom heritageshm.viz import apply_theme\n\napply_theme(context='notebook')'''

nb01 = "01_Data_Quality_and_Gaps.ipynb"
nb01_t = '''from heritageshm.diagnostics import characterize_gaps'''
nb01_r = '''from heritageshm.diagnostics import characterize_gaps\nfrom heritageshm.viz import apply_theme\n\napply_theme(context='notebook')'''

nb02 = "02_Proxy_Validation_and_Lags.ipynb"
nb02_t = '''from heritageshm.diagnostics import shift_and_correlate'''
nb02_r = '''from heritageshm.diagnostics import shift_and_correlate\nfrom heritageshm.viz import apply_theme\n\napply_theme(context='notebook')'''

if os.path.exists(nb00): update_notebook(nb00, nb00_t, nb00_r)
if os.path.exists(nb01): update_notebook(nb01, nb01_t, nb01_r)
if os.path.exists(nb02): update_notebook(nb02, nb02_t, nb02_r)

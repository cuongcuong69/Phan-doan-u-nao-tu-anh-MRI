import json
import os

notebook_path = r"d:\Project Advanced CV\colab_project\Start_UNETR_Colab.ipynb"
log_path = r"d:\Project Advanced CV\colab_project\debug_log.txt"

with open(log_path, 'w', encoding='utf-8') as log:
    try:
        if not os.path.exists(notebook_path):
            log.write("Notebook not found!\n")
            exit(1)
            
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        found = False
        log.write("Scanning cells...\n")
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                source = cell['source']
                # Join lines to inspect
                full_source = "".join(source)
                log.write(f"Cell {i} source: {full_source[:50]}...\n")
                
                # Check for substring
                if "trainers/train_unetr3d_brats2020.py" in full_source:
                    log.write(f"--> Found target in Cell {i}\n")
                    
                    # Update source
                    new_source = [
                        "# Chạy Training\n",
                        "DRIVE_CKPT_PATH = os.path.join(BASE_DRIVE_PATH, 'checkpoints')\n",
                        "os.makedirs(DRIVE_CKPT_PATH, exist_ok=True)\n",
                        "print(f\"📂 Auto-save checkpoints to: {DRIVE_CKPT_PATH}\")\n",
                        "\n",
                        "!python trainers/train_unetr3d_brats2020.py --drive_path \"{DRIVE_CKPT_PATH}\""
                    ]
                    cell['source'] = new_source
                    found = True
                    break

        if found:
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=2, ensure_ascii=False)
            log.write("Notebook updated successfully.\n")
        else:
            log.write("Training cell not found!\n")

    except Exception as e:
        log.write(f"Error: {e}\n")

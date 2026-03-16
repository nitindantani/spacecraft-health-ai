import os, sys

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

for folder in ['models/lstm','models/cnn','models/transformer','models/gnn','pipeline','core']:
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith('.py'):
                path = os.path.join(root, f)
                print(f'\n\n===== {path} =====')
                try:
                    content = open(path, encoding='utf-8', errors='replace').read()
                    print(content)
                except Exception as e:
                    print(f'[ERROR reading {path}: {e}]')
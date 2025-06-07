import os
from collections import defaultdict

# Diretório base onde estão as pastas train, val, test
base_dir = "dataset"  # Substitua pelo caminho real se diferente

# Dicionário para armazenar as contagens
counts = defaultdict(lambda: defaultdict(int))

# Percorre todas as pastas (train, val, test) e suas subpastas (labels)
for split in os.listdir(base_dir):
    split_path = os.path.join(base_dir, split)
    
    if os.path.isdir(split_path):
        for label in os.listdir(split_path):
            label_path = os.path.join(split_path, label)
            
            if os.path.isdir(label_path):
                # Conta o número de arquivos (imagens) na pasta
                num_images = len([
                    f for f in os.listdir(label_path) 
                    if os.path.isfile(os.path.join(label_path, f))
                ])
                counts[split][label] = num_images

# Exibe os resultados formatados
for split, labels in counts.items():
    print(f"\n=== {split.upper()} ===")
    for label, count in labels.items():
        print(f"{label}: {count} imagens")
    print(f"Total em {split}: {sum(labels.values())} imagens")

# Total geral
total_imagens = sum(sum(labels.values()) for labels in counts.values())
print(f"\n=== TOTAL GERAL ===\n{total_imagens} imagens")
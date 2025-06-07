import os
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import shutil

# Configurações
CSV_PATH = "Data_Entry_2017.csv"
BASE_IMAGES_DIR = "archive"
OUTPUT_DIR = "dataset"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
SEED = 42

# Processar CSV
df = pd.read_csv(CSV_PATH)
print(f"Total de entradas no CSV: {len(df)}")

image_labels_map = {}
for _, row in df.iterrows():
    filename = row['Image Index']
    labels = row['Finding Labels'].split('|')
    image_labels_map[filename] = [label.strip().upper() for label in labels]

# Coletar imagens
all_image_paths = []
for folder in glob(os.path.join(BASE_IMAGES_DIR, "images_*", "images")):
    all_image_paths.extend(glob(os.path.join(folder, "*.*")))

print(f"Total de imagens encontradas: {len(all_image_paths)}")

# Criar estrutura de splits
unique_files = list(image_labels_map.keys())
train_files, temp_files = train_test_split(
    unique_files,
    test_size=(1 - TRAIN_RATIO),
    random_state=SEED
)

val_files, test_files = train_test_split(
    temp_files,
    test_size=TEST_RATIO/(VAL_RATIO + TEST_RATIO),
    random_state=SEED
)

# Função para processar os splits
def process_split(file_list, split_name):
    moved_files = set()
    for filename in file_list:
        # Encontrar o caminho original da imagem
        img_path = None
        for folder in glob(os.path.join(BASE_IMAGES_DIR, "images_*", "images")):
            path = os.path.join(folder, filename)
            if os.path.exists(path):
                img_path = path
                break
        
        if img_path is None:
            continue
            
        labels = image_labels_map[filename]
        
        # Mover para cada label (apenas se o arquivo ainda existir)
        if os.path.exists(img_path):
            for label in labels:
                dest_dir = os.path.join(OUTPUT_DIR, split_name, label)
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, filename)
                
                if not os.path.exists(dest_path):
                    shutil.move(img_path, dest_path)
                    moved_files.add(filename)
                    img_path = dest_path  # Atualiza para o novo local
                else:
                    print(f"Arquivo já existe: {dest_path}")
        else:
            print(f"Arquivo não encontrado: {img_path}")

# Processar cada split
process_split(train_files, "train")
process_split(val_files, "val")
process_split(test_files, "test")

print("\nConcluído! Estrutura criada:")
print(f"  - Pasta principal: {OUTPUT_DIR}")
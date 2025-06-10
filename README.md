# Codes-PIBIC-2025 - Organização de Dataset de Raios-X Torácicos

## 📌 Visão Geral
Repositório contendo scripts para organização automática do dataset [NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data) em uma estrutura padronizada para projetos de machine learning.

## 🗂 Dataset Original
- **Fonte**: [NIH Chest X-ray Dataset no Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data?resource=download&select=Data_Entry_2017.csv)
- **Tamanho**: 112,120 imagens
- **Labels**: 14 condições médicas + "No Finding"

## 🛠 Scripts Disponíveis

### 1. `separacao_dataset.py`
**Função**: Organiza o dataset em train/val/test com base nas labels.

**Funcionalidades**:
- Divide automaticamente em 70% train, 20% val, 10% test
- Lida com imagens multi-label (uma imagem pode pertencer a múltiplas pastas)
- Move (não copia) os arquivos para a nova estrutura
- Cria pastas com nomes padronizados em maiúsculas

### 📊 Distribuição Geral
| Split  | Quantidade | Percentual |
|--------|------------|------------|
| Train  | 78,483     | 70%        |
| Val    | 22,424     | 20%        |
| Test   | 11,213     | 10%        |
| **Total** | **112,120** | **100%**  |

## 🗂 Estrutura do Dataset Organizado

### Estrutura de Diretórios
```plaintext
dataset/
│
├── train/                   # 78,483 imagens (70%)
│   ├── ATELECTASIS/         # 2,958 imagens
│   ├── CARDIOMEGALY/        # 858 imagens
│   ├── CONSOLIDATION/       # 1,220 imagens
│   ├── EDEMA/               # 561 imagens
│   ├── EFFUSION/           # 4,513 imagens
│   ├── EMPHYSEMA/          # 869 imagens
│   ├── FIBROSIS/           # 700 imagens
│   ├── HERNIA/             # 102 imagens
│   ├── INFILTRATION/       # 10,968 imagens
│   ├── MASS/               # 2,898 imagens
│   ├── NO_FINDING/         # 42,167 imagens
│   ├── NODULE/             # 3,844 imagens
│   ├── PLEURAL_THICKENING/ # 2,165 imagens
│   ├── PNEUMONIA/          # 951 imagens
│   └── PNEUMOTHORAX/       # 3,709 imagens
│
├── val/                     # 22,424 imagens (20%)
│   ├── ATELECTASIS/         # 851 imagens
│   ├── CARDIOMEGALY/        # 214 imagens
│   ├── CONSOLIDATION/       # 370 imagens
│   ├── EDEMA/               # 134 imagens
│   ├── EFFUSION/           # 1,284 imagens
│   ├── EMPHYSEMA/          # 242 imagens
│   ├── FIBROSIS/           # 186 imagens
│   ├── HERNIA/             # 30 imagens
│   ├── INFILTRATION/       # 3,094 imagens
│   ├── MASS/               # 822 imagens
│   ├── NO_FINDING/         # 12,148 imagens
│   ├── NODULE/             # 1,128 imagens
│   ├── PLEURAL_THICKENING/ # 575 imagens
│   ├── PNEUMONIA/          # 285 imagens
│   └── PNEUMOTHORAX/       # 1,061 imagens
│
└── test/                    # 11,213 imagens (10%)
    ├── ATELECTASIS/         # 420 imagens
    ├── CARDIOMEGALY/        # 113 imagens
    ├── CONSOLIDATION/       # 168 imagens
    ├── EDEMA/               # 78 imagens
    ├── EFFUSION/           # 700 imagens
    ├── EMPHYSEMA/          # 99 imagens
    ├── FIBROSIS/           # 91 imagens
    ├── HERNIA/             # 19 imagens
    ├── INFILTRATION/       # 1,515 imagens
    ├── MASS/               # 425 imagens
    ├── NO_FINDING/         # 6,046 imagens
    ├── NODULE/             # 563 imagens
    ├── PLEURAL_THICKENING/ # 303 imagens
    ├── PNEUMONIA/          # 157 imagens
    └── PNEUMOTHORAX/       # 516 imagens



import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

# Configurações
DATASET_DIR = "../dataset"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_CLASSES = 15

# Preparar diretórios
train_dir = os.path.join(DATASET_DIR, "train")
val_dir = os.path.join(DATASET_DIR, "val")
test_dir = os.path.join(DATASET_DIR, "test")

# Listar classes
class_names = sorted(os.listdir(train_dir))
print("Classes detectadas:", class_names)

# Geradores de dados
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=class_names,
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=class_names,
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=class_names,
    shuffle=False
)

# Construir modelo VGG16
base_model = VGG16(
    weights='imagenet', 
    include_top=False, 
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compilar modelo
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_auc',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Treinar modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping, checkpoint]
)

# Carregar melhor modelo
model.load_weights('best_model.h5')

# Avaliar no conjunto de teste
test_loss, test_acc, test_auc = model.evaluate(test_generator)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# Prever classes para o conjunto de teste
print("\nGerando previsões para o conjunto de teste...")
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)  # Converter probabilidades em rótulos de classe

# Obter rótulos verdadeiros
y_true = test_generator.classes

# Gerar matriz de confusão
conf_mat = confusion_matrix(y_true, y_pred)

# Plotar matriz de confusão
plt.figure(figsize=(15, 12))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, 
            yticklabels=class_names)
plt.title('Matriz de Confusão', fontsize=16)
plt.ylabel('Rótulo Verdadeiro', fontsize=14)
plt.xlabel('Rótulo Predito', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Gerar relatório de classificação
class_report = classification_report(
    y_true, 
    y_pred, 
    target_names=class_names,
    digits=4
)

print("\nRelatório de Classificação:")
print(class_report)

# Salvar relatório em arquivo
with open('classification_report.txt', 'w') as f:
    f.write(class_report)

# Salvar modelo final
model.save('chest_xray_vgg16_model.h5')
print("Modelo salvo como 'chest_xray_vgg16_model.h5'")
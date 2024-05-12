import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt


IMAGE_SIZE=100
BATCH_SIZE=32
EPOCHS=50
CHANNELS=3

# carga de las imagenes del directorio, aleatoriza, setea el tamaño de las imagenes y el batchsize
dataset= tf.keras.preprocessing.image_dataset_from_directory(
    'crop_prediction',
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size= BATCH_SIZE,
)

# obtengo los class names
class_names= dataset.class_names

train_size = 0.8


# separa el dataset en 3, uno para entrenar el modelo, otro para testear y el ultimo para validar
def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

# randomizo y los cargo en memoria
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# creo mas datos invirtiendo las imagenes y girandolas
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])

# aplico los cambios de data_augmentation
train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y)
).prefetch(buffer_size=tf.data.AUTOTUNE)

# define el input shape
input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
# define la cantidad de clases
n_classes = 3

# define la red
model = models.Sequential([
    # Capa de reescalado para normalizar los valores de píxel a [0, 1]
    layers.Rescaling(1./255),
    
    # Capas convolucionales seguidas de capas de pooling para extraer características
    layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Capa de aplanamiento para convertir las características en un vector unidimensional
    layers.Flatten(),
    
    # Capa densa con activación ReLU
    layers.Dense(64, activation='relu'),
    
    # Capa de salida con activación softmax para la clasificación multiclase
    layers.Dense(n_classes, activation='softmax'),
])

# hace el build del modelo
model.build(input_shape=input_shape)

# summary del modelo
model.summary()

# compila el modelo
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# entrena el modelo usando el train dataset
# valida con el validation datset
model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=EPOCHS,
)

# usa el test dataset para testear el modelo
scores = model.evaluate(test_ds)

# guarda el modelo para futuro uso o entrenamiento
model.save('crops_model2.keras')


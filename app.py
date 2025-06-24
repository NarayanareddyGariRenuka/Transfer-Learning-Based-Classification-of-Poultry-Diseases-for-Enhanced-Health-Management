# Import necessary libraries
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os

# Define constants
TRAIN_DIR = 'F:/project folder python/dataset/train'
VALIDATION_DIR = 'F:/project folder python/dataset/validation'  
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
NUM_CLASSES = 3  # Adjust based on the number of disease classes
EPOCHS = 10

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'F:/project folder python/dataset/train',
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical'  
)

validation_generator = validation_datagen.flow_from_directory(
    'F:/project folder python/dataset/validation',  # ✅ Use your actual path
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical'
)
# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base layers
base_model.trainable = False

# Add custom layers
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0)
# Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[reduce_lr, early_stop]
)

# Fine-tune the model (optional)
# Unfreeze some layers and train with a lower learning rate

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation accuracy: {accuracy:.2f}')

# Save the model
model.save('poultry_disease_classification_model.keras')
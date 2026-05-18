from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory(
    r"C:\Users\Jitendra Modha\Desktop\Desktop\virtual\data",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    color_mode="grayscale",
    subset='training'
    )

val_generator = datagen.flow_from_directory( 
    r"C:\Users\Jitendra Modha\Desktop\Desktop\virtual\data", 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='categorical', 
    color_mode="grayscale",
    subset='validation' 
)

num_classes = train_generator.num_classes

def training_model(a,b):
    model=Sequential()
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=(224,224,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))  # NEW
    model.add(Dropout(0.5))   
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )
    history = model.fit( a, epochs=15, validation_data=b )
    model.save("gesture_model_final.h5")
    return history


history=training_model(train_generator, val_generator)          
  
print("Model trained successfully.")

plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
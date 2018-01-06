from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
from keras import backend as K
K.set_image_dim_ordering('th')


def get_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))    

    # Converts 3D feature maps to 1D feature vectors
    model.add(Flatten())  

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def from_weights():
    weights_file = "first_try.h5"
    model = get_model()
    model.load_weights(weights_file)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    # img = load_img("images_processed/validation/loss/111.jpg")
    img = load_img("images_processed/train/other-memes/171db73579cdca0559be6e6c5d8366fe.jpg")
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)
    preds = model.predict(x)


def train(epochs):
    model = get_model()
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    batch_size = 16

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=False)  # No horizontal flip because that's not how loss works

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            'images_processed/train',  # this is the target directory
            target_size=(150, 150),  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            'images_processed/validation',
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='binary')    

    model.fit_generator(
            train_generator,
            steps_per_epoch=2000 // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=800 // batch_size)
    model.save_weights('1_epoch.h5')  # always save your weights after training or during training


def main():
    train(10)
    # from_weights()

if __name__ == "__main__":
    main()

import tensorflow
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D
import ConvertImage
import sketchBook
from tensorflow import keras


def display_some_examples(examples, labels):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        index = np.random.randint(0, examples.shape[0]-1)
        img = examples[index]
        label = labels[index]
        plt.subplot(5, 5, i+1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(img, cmap='gray')
    plt.show()


# sequential tensorflow model
model = tensorflow.keras.Sequential(
    [
        Input(shape=(28, 28, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        GlobalAvgPool2D(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ]
)


# functional model
def functional_model():
    my_input = Input(shape=(28, 28, 1))

    x = Conv2D(32, (3, 3), activation='relu')(my_input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = GlobalAvgPool2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    func_model = tensorflow.keras.Model(inputs=my_input, outputs=x)

    return func_model


if __name__ == '__main__':
    '''(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # display_some_examples(x_train, y_train)

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    while True:
        model = functional_model()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
        model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.2)

        loss, accuracy = model.evaluate(x_test, y_test, batch_size=64)

        if accuracy > 0.95:
            model.save('NumModel.h5')
            break'''
    while True:
        model = keras.models.load_model('NumModel.h5')

        sketchBook.sketch()
        myImage = ConvertImage.Conversion()
        myImage = np.flip(myImage, 1)
        myImage = np.rot90(myImage)
        test1 = myImage.astype('float32') / 255
        test = np.expand_dims(test1, axis=-1)
        test = np.expand_dims(test, axis=0)

        num_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        model = keras.models.load_model('NumModel.h5')
        model.summary()
        predict = model.predict(test)

        plt.title(num_list[np.argmax(predict)])
        plt.imshow(test1, cmap='gray')
        plt.show()


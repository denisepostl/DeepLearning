import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ImageClassifier:
    """
    A class representing an image classifier using Convolutional Neural Networks (CNN).

    Attributes:
        train_data_dir (str): The directory path containing the training data.
        img_width (int): The width of the input images (default is 150 pixels).
        img_height (int): The height of the input images (default is 150 pixels).
        batch_size (int): The batch size used for training (default is 32).
        model (tensorflow.keras.models.Sequential): The CNN model for image classification.
    """

    def __init__(self, train_data_dir, img_width=150, img_height=150, batch_size=32):
        """
        Initializes an ImageClassifier object.

        Args:
            train_data_dir (str): The directory path containing the training data.
            img_width (int, optional): The width of the input images (default is 150 pixels).
            img_height (int, optional): The height of the input images (default is 150 pixels).
            batch_size (int, optional): The batch size used for training (default is 32).
        """
        self.train_data_dir = train_data_dir
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size
        self.model = self.build_model()  # Initializing the model upon instantiation

    def build_model(self):
        """
        Builds and compiles a Convolutional Neural Network (CNN) model for image classification.

        Returns:
            tensorflow.keras.models.Sequential: The constructed CNN model.
        """
        model = Sequential()
        # Adding Convolutional layers with MaxPooling to extract features
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_width, self.img_height, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        # Flattening the layers and adding Dense layers for classification
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(3, activation='softmax'))  # Output layer with softmax activation for multi-class classification
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Compiling the model
        return model

    def train(self, epochs=10):
        """
        Trains the image classifier model using the provided training data.

        Args:
            epochs (int, optional): The number of training epochs (default is 10).
        """
        train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        # Generating training data from directory with specified parameters
        train_generator = train_datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'  # Taking a subset for training from the data
        )
        # Generating validation data from directory with specified parameters
        validation_generator = train_datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'  # Taking a subset for validation from the data
        )
        # Training the model using the generated data
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // self.batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // self.batch_size
        )

def main():
    """
    The main function to initiate the training of the ImageClassifier.
    """
    train_data_directory = 'training_data'  # Directory containing training data
    image_classifier = ImageClassifier(train_data_directory)
    image_classifier.train(epochs=10)  # Initiating the training process with 10 epochs

if __name__ == "__main__":
    main()

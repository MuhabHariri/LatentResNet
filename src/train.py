import os
import tensorflow as tf
import numpy as np

from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping

from src.config import *
from src.data_pipeline import build_dataset
from src.autoencoder import LiteAE
from src.model_blocks import EncodDeepResNet_Model
from src.layers import remove_spatial_dropout
from src.model_blocks import mlp_head
from src.augmentations import plot_and_save_images



import os
from tensorflow.keras.callbacks import Callback

class SaveWeightsPerEpoch(Callback):
    def __init__(self, output_dir="Model_weights", model_name="model_weights"):
        super(SaveWeightsPerEpoch, self).__init__()
        self.output_dir = output_dir
        self.model_name = model_name
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        file_path = os.path.join(self.output_dir, f"{self.model_name}_epoch_{epoch + 1}.h5")
        self.model.save_weights(file_path)
        print(f"✅ Saved weights at: {file_path}")


def run_training():
    print("Checking GPU availability...")
    print(f"Num GPUs available: {len(tf.config.experimental.list_physical_devices('GPU'))}")
    print(f"TensorFlow Version: {tf.version.VERSION}")

    #Step 1: Load File Paths & Build Datasets
    print("Preparing datasets...")
    train_file_paths = tf.io.gfile.glob(os.path.join(TRAIN_DIR, "*", "*"))
    val_file_paths = tf.io.gfile.glob(os.path.join(VAL_DIR, "*", "*"))

    train_dataset = build_dataset(train_file_paths, augment_data=True)
    val_dataset = build_dataset(val_file_paths)

    train_samples = len(train_file_paths)
    val_samples = len(val_file_paths)
    train_steps_per_epoch = int(np.ceil(train_samples / BATCH_SIZE))
    val_steps_per_epoch = int(np.ceil(val_samples / BATCH_SIZE))

    print(f"Train samples: {train_samples} | Val samples: {val_samples} | "
          f"Train steps/epoch: {train_steps_per_epoch} | Val steps/epoch: {val_steps_per_epoch}")

    #Step 2: Distributed Training Strategy
    strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
    )
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    with strategy.scope():
        #Step 3: Load Encoder
        print("Loading and modifying autoencoder...")
        autoencoder = LiteAE(input_shape=INPUT_SHAPE)

        # Use layer index based on model summary
        encoder = models.Model(inputs=autoencoder.input, outputs=autoencoder.layers[15].output)
        encoder.load_weights(ENCODER_WEIGHTS_PATH)

        encoder = remove_spatial_dropout(encoder)

        # Optional: Freeze encoder
        # for layer in encoder.layers:
        #     layer.trainable = False

        #Step 4: Create Classification Model
        inputs = tf.keras.Input(shape=INPUT_SHAPE)
        x = encoder(inputs)
        outputs = EncodDeepResNet_Model(x, num_classes=NUM_CLASSES)
        model = models.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

    model.summary()

    #Step 5: Callbacks
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    save_weights_callback = SaveWeightsPerEpoch(output_dir="Model_weights", model_name="final_model")


    # # plot
    # for image_batch, label_batch in train_dataset.take(1):
    #     plot_and_save_images(image_batch.numpy(), label_batch.numpy(),EPOCHS)


    #Step 6: Training Loop
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        steps_per_epoch=train_steps_per_epoch,
        validation_steps=val_steps_per_epoch,
        callbacks=[early_stopping, save_weights_callback]
    )

    # Save model
    model.save("models/final_classifier_model.h5")
    print("Model training complete and saved to models/")


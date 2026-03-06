import os

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def train_autoencoder(model, training_imgs, config):
    """
    Splits data, compiles model, and runs autoencoder training.

    Parameters
    ----------
    model : keras.Model
        Autoencoder model.
    training_imgs : np.ndarray
        Training images (including augmentations).
    config : dict
        Training configuration.

    Returns
    -------
    tuple
        (trained_model, history, validation_data)
    """
    train_data, validation_data = train_test_split(
        training_imgs,
        test_size=config.get('test_size', 0.2),
        random_state=config.get('random_state', 26),
    )

    model.compile(loss="mse", optimizer="adam")

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        os.path.join('artifacts', 'checkpoints', config['name'], '{epoch:02d}-{val_loss:.5f}.weights.h5'),
        save_best_only=True,
        verbose=1,
        save_weights_only=True,
    )

    callbacks = [checkpoint, early_stopping]
    callbacks.extend(config.get('callbacks', []))

    history = model.fit(
        train_data,
        train_data,
        validation_data=(validation_data, validation_data),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=callbacks,
    )
    return model, history, validation_data

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import logging
from flask import jsonify

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.regularizers import l2

def train(data):
    # Features and target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Split the data into train (60%), validation (20%), and test (20%) sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Learning rate scheduler function
    def lr_scheduler(epoch, lr):
        if epoch > 10:
            lr = lr * 0.9  # Reduce learning rate by 10% after 10 epochs
        return lr

    # Build a neural network model
    model = Sequential()
    model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu', kernel_regularizer=l2(0.001)))  # L2 regularization
    model.add(Dropout(0.3))  # Dropout layer to prevent overfitting
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

    # Compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Early stopping and learning rate scheduler callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler_cb = LearningRateScheduler(lr_scheduler)

    # Train the model using validation data
    history = model.fit(X_train_scaled, y_train,
                        epochs=100,
                        batch_size=32,
                        validation_data=(X_val_scaled, y_val),
                        callbacks=[early_stopping, lr_scheduler_cb],
                        verbose=1)

    # Save the model
    try:
        model_repo = os.environ['MODEL_REPO']
        if model_repo:
            file_path = os.path.join(model_repo, "heart_disease_model.h5")
            model.save(file_path)
            logging.info("Saved the model to the location: " + model_repo)
        else:
            raise KeyError
    except KeyError:
        model.save("heart_disease_model.h5")
        logging.info("The model was saved locally.")

    # Evaluate the model on the test set (final evaluation)
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
    logging.info(f"Test accuracy: {test_acc:.4f}")

    # Predictions
    y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)

    # Return the evaluation metrics
    text_out = {
        "accuracy": test_acc,
        "loss": test_loss,
    }
    logging.info(text_out)
    print(text_out)
    return jsonify(text_out), 200
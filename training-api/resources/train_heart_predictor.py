import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib
import logging
from flask import jsonify

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

def train(data):
    # Features and target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize StratifiedKFold for cross-validation on training data
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Collect results
    accuracies = []

    # Cross-validation loop on the training set
    for train_index, val_index in kf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

        # Standardize the data using training fold
        scaler = StandardScaler()
        X_fold_train_scaled = scaler.fit_transform(X_fold_train)
        X_fold_val_scaled = scaler.transform(X_fold_val)

        # Build the neural network model
        model = Sequential()
        model.add(Dense(128, input_dim=X_fold_train_scaled.shape[1], activation='relu', kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

        # Compile the model
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train the model using validation data
        history = model.fit(X_fold_train_scaled, y_fold_train,
                            epochs=200,
                            batch_size=16,
                            validation_data=(X_fold_val_scaled, y_fold_val),
                            callbacks=[early_stopping],
                            verbose=1)

        # Evaluate the model on validation set
        val_loss, val_acc = model.evaluate(X_fold_val_scaled, y_fold_val, verbose=1)
        accuracies.append(val_acc)

    # Final evaluation on the test set
    # Standardize the test set using the full training data scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Re-train the model on the entire training data
    model.fit(X_train_scaled, y_train, epochs=200, batch_size=16, verbose=0)

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)

    # Use the model to predict on the test set
    y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)

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
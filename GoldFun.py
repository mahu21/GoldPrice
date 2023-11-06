from pathlib import Path
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import numpy as np
from datetime import datetime

IMAGES_PATH = Path() / "images" 
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

#code to save the figures as high-res PNGs for the book
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
def negloglik(y, y_pred):
    return -y_pred.log_prob(y)    
    
def plot_training_metrics(history):
      
    # Plot training and validation losses
    s1=8
    s2=3
    plt.figure(figsize=(s1, s2))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    
    # Plot training and validation MAE (mean absolute error)
    plt.figure(figsize=(s1, s2))
    plt.plot(history['mae'], label='Training MAE')
    plt.plot(history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Training and Validation MAE')
    plt.legend()
    
    
    # Calculate RMSE from MSE
    train_rmse = np.sqrt(history['mse'])
    val_rmse = np.sqrt(history['val_mse'])
    plt.figure(figsize=(s1, s2))
    plt.plot(train_rmse, label='Training RMSE')
    plt.plot(val_rmse, label='Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Training and Validation RMSE')
    plt.legend()
    
    
    plt.figure(figsize=(s1, s2))
    plt.plot(history['accuracy'], label='Training accuray')
    plt.plot(history['val_accuracy'], label='Validation accuray')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation accuray')
    plt.legend()

    # Plot training and validation R-squared
    plt.figure(figsize=(s1, s2))
    plt.plot(history['r_squared'], label='Training R-squared')
    #plt.plot(history['val_r_squared'], label='Validation R-squared')
    plt.ylim(-1,1.1)
    plt.xlabel('Epoch')
    plt.ylabel('R-squared')
    plt.title('Training R-squared')
    plt.legend()
    plt.show()    
    
    # Plot training and validation R-squared
    plt.figure(figsize=(s1, s2))
    plt.plot(history['val_r_squared'], label='Validation R-squared')
    plt.xlabel('Epoch')
    plt.ylabel('R-squared')
    plt.title('Validation R-squared')
    plt.legend()
    plt.show() 
    
def r2_score(y_true, y_pred):
    y_diff = tf.reduce_sum(tf.square(y_true-y_pred))
    y_square = tf.reduce_sum(tf.square(y_true-tf.reduce_mean(y_true)))
    return 1 - (y_diff/y_square)
# Define custom R-squared metric
def r_squared(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ( SS_res/(SS_tot + K.epsilon()) )

def dateindex(date_start,daten):
    # Convert the date_start string to a datetime object
    #target_date = datetime.strptime(date_start, "%Y-%m-%d %H:%M:%S")
    
    # Find the index of the target date in the data list
    try:
        index = daten.index(date_start)
        print("Index of the target date:", index)
    except ValueError:
        print("Target date not found in the data.")
    return(index)


    
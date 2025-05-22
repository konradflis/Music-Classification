
import matplotlib.pyplot as plt

def plot_training_history(history):
    epochs = range(len(history.history['loss']))

    plt.figure(figsize=(16, 5))

    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history.history['loss'], label='Train Loss')
    plt.plot(epochs, history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history.history['accuracy'], label='Train Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Precision (find key dynamically)
    precision_keys = [k for k in history.history.keys() if "precision" in k and not k.startswith("val")]
    val_precision_keys = [k for k in history.history.keys() if "val_precision" in k]

    if precision_keys and val_precision_keys:
        plt.subplot(1, 3, 3)
        plt.plot(epochs, history.history[precision_keys[0]], label='Train Precision')
        plt.plot(epochs, history.history[val_precision_keys[0]], label='Val Precision')
        plt.title('Precision')
        plt.xlabel('Epochs')
        plt.ylabel('Precision')
        plt.legend()

    plt.tight_layout()
    plt.show()

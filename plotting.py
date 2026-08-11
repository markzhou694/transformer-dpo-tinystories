import matplotlib.pyplot as plt


def plot_train_test_loss_curves(train_losses, test_losses, log_interval):
  #(Added Functionality) Visualizes the training loss over time.
#It compares the training loss against the final test loss for all trained models.
#The resulting graph is saved as a PNG file.
    """
    Plots training and test loss curves for multiple models on the same figure.

    Args:
        train_losses (dict): {model_name: [list of train loss values]}
        test_losses (dict): {model_name: scalar test loss}
        log_interval (int): Number of steps between logging losses
    """
    plt.figure(figsize=(12, 7))

    # Define consistent colors for each model
    colors = {
        "kgram_mlp_seq": (0.2, 0.4, 0.8),   # blue
        "lstm_seq": (0.9, 0.5, 0.1),        # orange
        "transformer": (0.2, 0.7, 0.3),     # green
    }

    for model_name in train_losses.keys():
        color = colors.get(model_name, (0.1, 0.7, 0.3))
        steps = [i * log_interval for i in range(1, len(train_losses[model_name]) + 1)]

        # Plot training curve
        plt.plot(
            steps,
            train_losses[model_name],
            label=f"{model_name} - Train",
            color=color
        )

        # Plot test loss as dashed horizontal line
        if model_name in test_losses:
            plt.axhline(
                y=test_losses[model_name],
                color=color,
                linestyle='--',
                alpha=0.6,
                label=f"{model_name} - Test"
            )

    plt.title("Train vs Test Loss", fontsize=16)
    plt.xlabel("Training Steps", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend()
    plt.grid(True, which='both', linestyle=':', linewidth=0.7)
    plt.yscale('log')  # useful when loss spans several orders of magnitude
    plt.tight_layout()
    plt.savefig("training_and_test_loss_plot.png", dpi=300)
    print("Saved combined train/test loss plot to 'training_and_test_loss_plot.png'")

import numpy as np
from matplotlib import pyplot as plt
from pyod.utils import standardizer


def clip(all_x, x_train):
    all_x_clipped = np.copy(all_x)

    # Clip the values in all_x based on the upper and lower limits from x_train
    for col in range(all_x_clipped.shape[1]):  # Iterate through each column
        lower_limit = min(
            x_train[:, col]
        )  # Get the lower limits from Y for the current column
        upper_limit = max(
            x_train[:, col]
        )  # Get the upper limits from Y for the current column

        # Clip or cap the values in the current column of x_train
        all_x_clipped[:, col] = np.clip(all_x_clipped[:, col], lower_limit, upper_limit)

    return all_x_clipped


def plot_points(x_train_original, x_train, x_test, x_gen, y_gen, figures_path):
    # Define a list of colours for each label
    colours = ["blue", "yellow"]

    # Create a figure with four subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot the first subplot with x_train data
    ax1.plot(x_train_original[:, 3], x_train_original[:, 5], ".", alpha=0.5)
    ax1.set_xlim([-6, 6])
    ax1.set_ylim([-6, 6])
    ax1.set_title("X Train Original")
    ax1.grid()

    # Plot the fourth subplot with x_test data
    ax2.plot(x_test[:, 3], x_test[:, 5], ".", alpha=0.5)
    ax2.set_xlim([-6, 6])
    ax2.set_ylim([-6, 6])
    ax2.set_title("X Test Normalized")
    ax2.grid()

    # Plot the third subplot with x_train_norm data
    ax3.plot(x_train[:, 3], x_train[:, 5], ".", alpha=0.5)
    ax3.set_xlim([-6, 6])
    ax3.set_ylim([-6, 6])
    ax3.set_title("X train Normalized")
    ax3.grid()

    # Plot the second subplot with x generated training data and colour them by y_gen labels
    ax4.scatter(x_gen[:, 3], x_gen[:, 5], c=[colours[int(i)] for i in y_gen], alpha=0.5)
    ax4.set_xlim([-6, 6])
    ax4.set_ylim([-6, 6])
    ax4.set_title("X Generated Train Normalized")
    ax4.grid()

    # Save the figure to the specified path
    plt.savefig(figures_path)


def plot_importances(n_features, importances, feature_names, save_path):
    # isolate the dataset name is
    dataset_name = save_path.stem
    dataset_name = dataset_name[:-4]

    # convert the feature_names to an array if it is not already
    feature_names = np.array(feature_names)

    # sort the importances and feature names in descending order
    sorted_indices = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_indices]
    sorted_feature_names = feature_names[sorted_indices]

    # select the top n importances and feature names
    top_n_importances = sorted_importances[:n_features]
    top_n_feature_names = sorted_feature_names[:n_features]

    # create a figure
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_n_importances)), top_n_importances, align='center')
    plt.yticks(range(len(top_n_importances)), top_n_feature_names)
    plt.xlabel('Relative Importance')
    plt.title(f'Top {n_features} Important Features of Decision Tree for {dataset_name}')
    plt.savefig(save_path)
    plt.close()

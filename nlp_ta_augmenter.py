# Import libraries
import random
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from scipy.interpolate import griddata

from textattack.augmentation import Augmenter
from textattack.transformations import WordSwapQWERTY, WordSwapRandomCharacterDeletion, CompositeTransformation
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification


# Build customized augmenter
def create_custom_augmenter(pct_words_to_swap=0.1, transformations_per_example=1):
    """
    Creates a customized TextAttack augmenter with specified parameters.
    """
    # Set up transformations
    transformation = CompositeTransformation([
        WordSwapRandomCharacterDeletion(),
        WordSwapQWERTY()
    ])
    
    # Set up constraints
    constraints = [
        RepeatModification(),
        StopwordModification()
    ]
    
    # Create and return the augmenter with specified parameters
    ca_augmenter = Augmenter(
        transformation=transformation,
        constraints=constraints,
        pct_words_to_swap=pct_words_to_swap,
        transformations_per_example=transformations_per_example
    )
    
    return ca_augmenter


# Data Augmentation
def augment_data(data, target, augmenter, augmentation_chance=0.5):
    augmented_data, augmented_labels = [], []
    for text, label in zip(data, target):
        augmented_data.append(text)
        augmented_labels.append(label)
        if random.random() < augmentation_chance:
            augmented_texts = augmenter.augment(text)
            augmented_data.extend(augmented_texts)
            augmented_labels.extend([label] * len(augmented_texts))
    return augmented_data, augmented_labels



def plot_heatmap(df):

    # Pivot the DataFrame to create a matrix suitable for heatmap plotting
    pivot_table = df.pivot(index="pct_words_to_swap", columns="transformations_per_example", values="accuracy")

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2%", cmap="viridis")
    plt.title("Model Accuracy by pct_words_to_swap and transformations_per_example")
    plt.xlabel("Transformations per Example")
    plt.ylabel("Percentage of Words to Swap")
    plt.show()

def plot_3d_surface(df):
    # Prepare data for interpolation
    points = df[['pct_words_to_swap', 'transformations_per_example']].values
    values = df['accuracy'].values

    # Create grid
    x_grid, y_grid = np.meshgrid(np.linspace(df['pct_words_to_swap'].min(), df['pct_words_to_swap'].max(), 20),
                                np.linspace(df['transformations_per_example'].min(), df['transformations_per_example'].max(), 20))

    # Interpolate z values over the grid
    z_grid = griddata(points, values, (x_grid, y_grid), method='cubic')

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x_grid, y_grid, z_grid, cmap='viridis', edgecolor='none')
    ax.set_xlabel('pct_words_to_swap')
    ax.set_ylabel('transformations_per_example')
    ax.set_zlabel('Accuracy')
    ax.set_title('Model Accuracy Surface Plot')
    fig.colorbar(surf, shrink=0.5, aspect=5)  

    plt.show()





import matplotlib.pyplot as plt
import seaborn as sns


def plot_categorical_count(df, column, top_n=None, hue=None, rotate_xticks=True):
    """
    Plots a countplot for a categorical column.

    Args:
        df (pd.DataFrame): The data.
        column (str): The column to plot.
        top_n (int, optional): Number of top categories to show. If None, show all.
        hue (str, optional): Column to use for color separation.
        rotate_xticks (bool): Whether to rotate x-axis labels.
    """
    if top_n:
        top_categories = df[column].value_counts().nlargest(top_n).index
        plot_data = df[df[column].isin(top_categories)]
    else:
        plot_data = df

    plt.figure(figsize=(18, 5))
    sns.countplot(data=plot_data, x=column, hue=hue, order=plot_data[column].value_counts().index)
    plt.title(f'Countplot of {column}')
    if rotate_xticks:
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_numeric_distribution(df, column, plot_type='hist', hue=None, bins=30, kde=True):
    """
    Plots a numeric distribution using different plot types.

    Args:
        df (pd.DataFrame): The data.
        column (str): The column to plot.
        plot_type (str): Type of plot: 'hist', 'kde', 'box', or 'violin'.
        hue (str, optional): Column to use for color separation.
        bins (int): Number of bins for histograms.
        kde (bool): Whether to include KDE on histograms.
    """
    plt.figure(figsize=(8, 5))

    if plot_type == 'hist':
        sns.histplot(data=df, x=column, hue=hue, bins=bins, kde=kde)
        plt.title(f'Histogram of {column}')
    elif plot_type == 'kde':
        sns.kdeplot(data=df, x=column, hue=hue, fill=True)
        plt.title(f'KDE Plot of {column}')
    elif plot_type == 'box':
        sns.boxplot(data=df, x=hue, y=column) if hue else sns.boxplot(data=df, y=column)
        plt.title(f'Box Plot of {column}')
    elif plot_type == 'violin':
        sns.violinplot(data=df, x=hue, y=column) if hue else sns.violinplot(data=df, y=column)
        plt.title(f'Violin Plot of {column}')
    else:
        raise ValueError("Invalid plot_type. Choose from 'hist', 'kde', 'box', 'violin'.")

    plt.tight_layout()
    plt.show()

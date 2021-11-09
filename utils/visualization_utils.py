import matplotlib.pyplot as plt
import seaborn as sns

def plot_models_results(data, x, y, hue, model_name):
    
    sns.set_style(style='darkgrid')

    sns.barplot(x=x, y=y, hue=hue, data = data)

    plt.ylabel("R2 value")
    plt.xlabel("dataset type")
    plt.title(f"{model_name} scores")
    
    plt.savefig(f"./plots/{model_name}_scores.png")

    plt.clf()

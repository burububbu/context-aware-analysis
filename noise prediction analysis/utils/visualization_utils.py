import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_loss(losses, params, loss_type, set_type='', prepr_type='', to_save=True):
    plt.clf()
    plt.plot(list(range(params['n_epochs'])), losses)

    plt.ylabel("mse loss")
    plt.xlabel("epochs")

    params_to_string = "E{0}_HS{1}_NL_{2}_BS_{3}_LR{4}_G{5}_DROP{6}".format(
        params['n_epochs'],
        params['hs'],
        params['n_layers'],
        params['b_size'],
        str(params['l_rate']).replace('.', '_'),
        str(params['gamma']).replace('.', '_'),
        str(params['dropout']).replace('.', '_'))

    plt.title(f"{params_to_string} training {loss_type}")

    if to_save:
        save(f"./plots/{set_type}_{prepr_type}_losses_{params_to_string}_.png")


# init useful functions
def grouped_bar_plot(x, y, hue, data, model_name, legend_pos='lower right'):
    # plt.figure(dpi=dpi)

    sns.set_style(style='darkgrid')
    sns.barplot(x=x, y=y, hue=hue, data=data)
    plt.legend(loc=legend_pos)

    plt.ylabel(f"{y.replace('_', ' ')} value")
    plt.xlabel(x.replace('_', ' '))
    plt.title(f"{model_name} scores")


def train_test_bar_plot(train_values, test_values, groups, score_type, title):
    fig = plt.figure()
    X = np.arange(0, len(groups)/2, 0.5)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(X + 0.00, train_values, width=0.20)
    ax.bar(X + 0.20, test_values, width=0.20)
    plt.xticks(X + 0.10, groups)

    ax.legend(labels=['Train', 'Test'])
    plt.xlabel('Type')
    plt.ylabel(score_type)

    plt.title(title)


def save(title):
    plt.savefig(title)
    plt.clf()

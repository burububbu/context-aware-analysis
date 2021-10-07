import  matplotlib.pyplot as plt

def plot(x_label, y_label, title, file_name):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig("./plots/{}.png".format(file_name))
    # plt.show()
    plt.clf()

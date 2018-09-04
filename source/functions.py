import numpy as np

def plot_scores(ax, x_axis, y_axis, n_split, color, title, label, alpha=0.1, legend_loc='best', plot_range=True):

    # plot the range of each score value
    if plot_range:
        ax.fill_between(x_axis, y_axis.min(axis=1), y_axis.max(axis=1), alpha=alpha, color=color)
        for i in range(n_split):
            ax.scatter(x_axis, y_axis[:, i], c=color, marker='+')


    # plot the main trending curve for each alpha value with labels
    y_axis_means = np.mean(y_axis, axis=1)
    ax.plot(x_axis, y_axis_means, label=label, c=color)


    # attach corresponding value for each point (commented here since all words get clustered together)
    # for i, x_coord in enumerate(x_axis):
    #     ax.annotate('{0:.2f}'.format(y_axis_means[i]), (x_coord, y_axis_means[i]))


    ax.set_title(title)
    ax.legend(loc=legend_loc)
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_performances(epoch, performances, ev, pl_dir, m_name):
    epochs = range(1, epoch + 1)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.xaxis.set_label_coords(0.1, -0.2)
    ax1.plot(epochs, performances['train_loss'], 'g', label='Training loss')
    if ev is not None:
        ax1.plot(epochs, performances['eval_loss'], 'b', label='Validation loss')
        ax1.set_title('Training and Validation loss')
    else:
        ax1.set_title('Training loss')
    ax1.set(xlabel='Epochs', ylabel='Loss')
    ax1.xaxis.set_label_coords(0.1, -0.2)
    ax1.legend()
    ax2.plot(epochs, performances['train_acc'], 'g', label='Training accuracy')
    if ev is not None:
        ax2.plot(epochs, performances['eval_acc'], 'b', label='Validation accuracy')
        ax2.set_title('Training and Validation accuracy')
    else:
        ax2.set_title('Training accuracy')
    ax2.set(xlabel='Epochs', ylabel='Accuracy')
    ax2.xaxis.set_label_coords(0.1, -0.2)
    ax2.legend()

    plt.tight_layout()

    fig_dir = os.path.join(pl_dir, m_name + '_perf_training.png')
    plt.savefig(fig_dir)
    # This will clear the first plot
    plt.close('all')

    return


def plot_confusion_matrix(p_test, conf_mat, cls_names, plot_dir, model_name, only_mat=False):
    if not only_mat:
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(322)
        ax3 = plt.subplot(324)
        ax4 = plt.subplot(326)
    else:
        ax1 = plt.subplot(111)

    num_cls = len(cls_names)
    # plot 1 confusion matrix
    _ = ax1.imshow(conf_mat)  # x axis = real class, y axis = predicted class
    ax1.set_xticks(np.arange(num_cls))
    ax1.set_yticks(np.arange(num_cls))
    ax1.set_xticklabels(cls_names)
    ax1.set_yticklabels(cls_names)
    ax1.set_ylabel('Real Class', fontweight='bold', fontsize=9)
    ax1.xaxis.set_label_coords(.5, 1.15)
    ax1.xaxis.tick_top()
    ax1.set_xlabel('Predicted Class', fontweight='bold', fontsize=9)
    ax1.xaxis.set_label_position('top')
    ax1.tick_params(axis='y', which='both', rotation=90)

    # Loop over data dimensions and create text annotations.
    for i in range(num_cls):
        for j in range(num_cls):
            _ = ax1.text(j, i, round(conf_mat[i, j], 3),
                         ha="center", va="center", color="w")

    if not only_mat:
        # plot 2 recall
        ax2.set_title("Recall", fontweight='bold', fontsize=12)
        _ = ax2.imshow(np.atleast_2d(p_test['recall']))
        for i, c in enumerate(cls_names):
            _ = ax2.text(i, 0, str(round(p_test['recall'][i], 2)), ha="center", va="center", color="w")
        ax2.tick_params(top=False, bottom=False, left=False, right=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)

        # plot 3 precision
        ax3.set_title("Precision", fontweight='bold', fontsize=12)
        _ = ax3.imshow(np.atleast_2d(p_test['precision']))
        for i, c in enumerate(cls_names):
            _ = ax3.text(i, 0, str(round(p_test['precision'][i], 2)), ha="center", va="center",
                         color="w")
        ax3.tick_params(top=False, bottom=False, left=False, right=False)
        plt.setp(ax3.get_xticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)

        # plot 4 f1
        ax4.set_title("F1 scores", fontweight='bold', fontsize=12)
        _ = ax4.imshow(np.atleast_2d(p_test['f1_score']))
        for i, c in enumerate(cls_names):
            _ = ax4.text(i, 0, str(round(p_test['f1_score'][i], 2)), ha="center", va="center",
                         color="w")

        ax4.tick_params(top=False, bottom=False, left=False, right=False)
        plt.setp(ax4.get_xticklabels(), visible=False)
        plt.setp(ax4.get_yticklabels(), visible=False)

    fig_dir = os.path.join(plot_dir, '{}_conf_mtrx.png'.format(model_name))
    plt.savefig(fig_dir)
    plt.close('all')
    # plt.show()

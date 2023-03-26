import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torchvision.transforms as T
from torchvision.utils import make_grid


attributes = [
"balancing_elements",
"color_harmony",
"content",
"depth_of_field",
"light",
"motion_blur",
"object",
"repetition",
"rule_of_thirds",
"symmetry",
"vivid_color",
]
to_pil = T.ToPILImage()


def display_aesthetics(x, y, y_pred, interval=[-0.2, 0.2], score=True):

    neg = interval[0]
    pos = interval[1]

    if score:
        y = y[:-1]
        y_pred = y_pred[:-1]

    # Initiate figure
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    # Picture
    ax0.imshow(to_pil(x))
    ax0.set_axis_off()

    # True results
    if score:
        ax1.set_title("Score of the real data: {:.2f}".format(y[-1]))
    else:
        "Real Data"
    states = np.zeros_like(y, dtype=bool)
    states[y >= pos] = True
    states[y <= neg] = True

    value_tag = [(v, t) for v, t in zip(attributes, y)]
    checkboxes_true = mpl.widgets.CheckButtons(ax1, value_tag, actives=states)

    choices = {0: "gray", -1: "red", 1: "green"}
    colors = np.zeros_like(y, dtype=int)
    colors[y >= pos] = 1
    colors[y <= neg] = -1
    [rec.set_facecolor(choices[c]) for rec, c in zip(checkboxes_true.rectangles, colors)]

    # Predicted Results
    if score:
        ax2.set_title("Score of the predicted data: {:.2f}".format(y_pred[-1]))
    else:
        "Predicted Data"
    states = np.zeros_like(y_pred, dtype=bool)
    states[y_pred >= pos] = True
    states[y_pred <= neg] = True

    value_tag = [(v, t) for v, t in zip(attributes, y_pred)]
    checkboxes_true = mpl.widgets.CheckButtons(ax2, value_tag, actives=states)

    choices = {0: "gray", -1: "red", 1: "green"}
    colors = np.zeros_like(y_pred, dtype=int)
    colors[y_pred >= pos] = 1
    colors[y_pred <= neg] = -1
    [rec.set_facecolor(choices[c]) for rec, c in zip(checkboxes_true.rectangles, colors)]

    # Plot
    plt.tight_layout()
    plt.show()

def show_example(img, label):
    print(label)
    plt.imshow(img.permute(1, 2, 0))

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=8).permute(1, 2, 0))
        break

def round_labels(labels):

    for index, value in enumerate(labels):
        if np.logical_and(value >= 0, value < 0.2):
            labels[index] = 0
        elif np.logical_and(value >= 0.2, value < 0.4):
            labels[index] = 1
        elif np.logical_and(value >= 0.4, value < 0.6):
            labels[index] = 2
        elif np.logical_and(value >= 0.6, value < 0.8):
            labels[index] = 3
        elif np.logical_and(value >= 0.8, value <= 1):
            labels[index] = 4

    return labels

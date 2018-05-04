from matplotlib import pyplot as plt


def ceildiv(a, b):
    return -(-a // b)


def plots_from_files(imspaths, figsize=(10, 5), rows=1, titles=None, maintitle=None):
    """Plots images given image files.
    Arguments:
        im_paths (list): list of paths
        figsize (tuple): figure size
        rows (int): number of rows
        titles (list): list of titles
        maintitle (string): main title
    """
    f = plt.figure(figsize=figsize)
    if maintitle is not None:
        plt.suptitle(maintitle, fontsize=16)
    for i in range(len(imspaths)):
        sp = f.add_subplot(rows, ceildiv(len(imspaths), rows), i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        img = plt.imread(imspaths[i])
        plt.imshow(img)
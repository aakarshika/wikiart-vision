from matplotlib import pyplot as plt
from PIL import Image

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

# genre 0

genre_0 = ['/home/bhavika/wikiart/Minimalism/walter-darby-bannard_sistene-1961.jpg', '/home/bhavika/wikiart/Abstract_Expressionism/vasile-dobrian_i-have-tried.jpg',
           '/home/bhavika/wikiart/Abstract_Expressionism/theodoros-stamos_flight-of-the-spectre-1949.jpg',
           '/home/bhavika/wikiart/Abstract_Expressionism/sam-francis_untitled-1947.jpg']

plots_from_files(genre_0)
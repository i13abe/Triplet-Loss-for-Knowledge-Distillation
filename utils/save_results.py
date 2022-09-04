import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.animation as anm
from sklearn.neighbors import NearestNeighbors


def save_info(
    info,
    filename="info",
):
    """
    Save the information to txt and csv file.
    Args:
        info (dict): information. "key" denotes information tag, "value" denotes information value.
        filename (str): output file name
    """
    info_str = ""
    for k, v in info.items():
        info_str += k + ":" + str(v) + "\n"
    
    with open(filename + '.txt', mode='w') as f:
        f.write(info_str)
    
    with open(filename + '.csv', 'w') as f:  
        writer = csv.writer(f)
        for k, v in info.items():
            writer.writerow([k, v])
            

def plot_graph(
    data_dict,
    y_lim=1.0,
    legend=True,
    filename='graph',
):
    """
    Plot graph.
    data_dict has label name and data.
    e.g.:
        data_dict = {"label1":[...],
                     "label2":[...],...}
    Args:
        data_dict (dict of list): data list. "key" denotes the information of data, "value" denotes the seaquence of data.
        legend (bool): on legends. Default is True
        y_lim (float): The upper limitation of y axis.
        filename (str): output file name
    """
    plt.figure(figsize=(6, 6))

    keys = data_dict.keys()
    for key in keys:
        plt.plot(
            range(len(data_dict[key])),
            data_dict[key],
            label=key,
        )
    plt.ylim(0, y_lim)
    if legend:
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0,
            ncol=1,
        )
    plt.savefig(filename+".png", bbox_inches='tight')
    plt.show()
    plt.close()

    
def plot_images(
    images,
    num_row,
    num_col,
    filename='images',
):
    """
    show images.
    e.g. of images:
        images = [[ndarray1], [ndarray2], ...]
    Args:
        images (list of numpy image): images list
        num_row (int): number of row
        num_col (int): number of column
        filename (str): output file name
    """
    fig,axes = plt.subplots(nrows=num_row,ncols=num_col,figsize=(10,8))
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    for i in range(num_row):
        for j in range(num_col):
            image = images[i*num_col+j]
            image[image<0.] = 0.
            if image.shape[-1] == 1:
                axes[i,j].imshow(image, cmap="gray")
            else:
                axes[i,j].imshow(image)
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])
    plt.savefig(filename+".png", bbox_inches='tight')
    plt.show()
    plt.close()
    
    
def plot_scatter(
    data_dict,
    alpha=0.5,
    s=5,
    legend=True,
    no_ticks=True,
    filename='scatter',
):
    """
    Plot scatter.
    data_dict has label name and 2d data.
    e.g.:
        data_dict = {"label1":[2d ndarray],
                     "label2":[2d ndarray],...}
    Args:
        data_dict (dict of list): data list. each data must be 2d. "key" denotes the information of data, "value" denotes the seaquence of data.
        alpha (float): The alpha of plot
        s (float): The size of plot
        legend (bool): on legends. Default is True
        no_ticks (bool): no ticks. Default is True
        filename (str): output file name
    """
    plt.figure(figsize=(6, 6))
    
    keys = data_dict.keys()
    for key in keys:
        plt.scatter(
            data_dict[key][:,0],
            data_dict[key][:,1],
            alpha=alpha,
            s=s,
            label=key,
        )
    if legend:
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0,
            ncol=1,
        )
    if no_ticks:
        plt.xticks(color="None")
        plt.yticks(color="None")
        plt.tick_params(length=0)
    plt.savefig(filename+".png", bbox_inches='tight')
    plt.show()
    plt.close()
    
    
def plot_im_scatter(
    data_dict,
    image_dict,
    alpha=0.7,
    zoom=0.3,
    no_ticks=True,
    filename='imScatter',
    max_class_samples=None,
):
    """
    Plot scatter with images.
    The data_dict has label name and 2d data.
    e.g.:
        data_dict = {"label1":[2d ndarray],
                     "label2":[2d ndarray],...}
    The image_dict has label name and images. Each images are corresponding to each data of data_dict.
    e.g.:
        image_dict = {"label1":[image1-1, image1-2,...],
                      "label2":[image2-1, image2-2,...],...}
    Args:
        data_dict (dict of list): Data list. Each data must be 2d.
                The "key" denotes the information of data, "value" denotes the seaquence of data.
        image_dict (dict of image list): Image list. Each image is numpy array.
                Each images are corresponding to each data of data_dict.
        alpha (float): The alpha of plot
        zoom (float): Zoom of image
        no_ticks (bool): No ticks. Defaults to True
        filename (str): Output file name. Defaults to 'imScatter'.
        max_class_samples (List[int], int or None): Max samples for visualization on each class. Defaults to None.
    """
    if max_class_samples is None:
        max_class_samples = []
        for data in data_dict.values():
            max_class_samples.append(len(data))
    
    if isinstance(max_class_samples, int):
        max_class_samples = [max_class_samples]*len(data_dict)

    assert len(max_class_samples) == len(data_dict),\
    f'The size of max_class_samples and data_dict is difference. '\
    f'max_class_samples is {len(max_class_samples)}, data_dict size is {len(data_dict)}.'
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    
    keys = data_dict.keys()
    num_classes = len(keys)
    for i, key in enumerate(keys):
        for data, image in zip(data_dict[key][:max_class_samples[i]], image_dict[key][:max_class_samples[i]]):            
            im = OffsetImage(
                image,
                zoom=zoom,
            )
            ab = AnnotationBbox(
                im,
                (data[0], data[1]),
                xycoords="data",
                frameon=True,
            )
            ab.patch.set_edgecolor(cm.jet(i/num_classes))
            ax.add_artist(ab)
            ax.plot(data[0], data[1], alpha=0)
    if no_ticks:
        plt.xticks(color="None")
        plt.yticks(color="None")
        plt.tick_params(length=0)
    plt.savefig(filename+".png", bbox_inches='tight')
    plt.show()
    plt.close()
        
    
def plot_histogram(
    data_dict,
    alpha=0.5,
    bins=50,
    ylim=500,
    legend=True,
    filename="histogram",
):
    """
    Plot histogram.
    data_dict has label name and data.
    e.g.:
        data_dict = {"label1":[1d ndarray],
                     "label2":[1d ndarray],...}
    Args:
        data_dict (dict of list): data list. each data must be 1d. "key" denotes the information of data, "value" denotes the freaquency.
        alpha (float): The alpha of histogram bin
        bins (float): the number of bins
        ylim (float): The limitation of y axis
        legend (bool): on legends. Default is True
        filename (str): output file name
    """
    plt.figure(figsize=(6, 6))
    
    keys = data_dict.keys()
    for key in keys:
        plt.hist(
            data_dict[key],
            alpha=alpha,
            bins=bins,
            label=key,
        )
    if legend:
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0,
            ncol=1,
        )
    plt.ylim(0, ylim)
    plt.savefig(filename+".png", bbox_inches='tight')
    plt.show()
    plt.close()

    
def plot_words(
    data_numpy,
    words,
    num_samples=5,
    base_word=None,
    filename="word",
):
    """
    plor words.
    e.g.
        data_numpy = [[x1, x2],
                      [y1, y2],...]
        words = ["x_word", "y_word",...]
    Args:
        data_numpy (2d ndarray): scatter ndarray
        words (list): words list
        num_samples (int): number of samples around base_word.
        base_word (None or str or list): base words.
        filename (str): file name
    Returns:
        detect_words (list of str): detected words
    """
    plt.figure(figsize=(6, 6))
    
    nbrs = NearestNeighbors(n_neighbors=num_samples).fit(data_numpy)
    dist, ind = nbrs.kneighbors(data_numpy)
    
    if intstance(base_word, str):
        base_word = [base_word]
    
    detect_words = []
    for word in base_word:
        n = words.index(word)
        n = ind[n]
        
        data_n = data_numpy[n]
        nearest_words = []
        for k in range(num_samples):
            nearest_words.append(words[n[k]])
            plt.scatter(
                data_n[k,0],
                data_n[k,1],
                alpha=0.0,
                color="black",
            )
            plt.annotate(
                words[n[k]],
                xy=(data_n[k,0], data_n[k,1]),
            )
        detected_words.append(nearest_words)
            
    plt.savefig(filename+".png", bbox_inches='tight')
    plt.show()
    plt.close()
    return detected_words
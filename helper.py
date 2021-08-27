import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.legend()
    plt.show(block=False)
    plt.pause(.1)


def unpack_tuple_list(tuple_list):
    res = []
    for element in tuple_list:
        if type(element) == tuple:
            temp1 = element[0]
            temp2 = element[1]
            res.append(temp1)
            res.append(temp2)
        else:
            res.append(element)
    return res
    
def add_flatten_lists(the_lists):
    result = []
    for _list in the_lists:
        result += _list
    result = unpack_tuple_list(result)
    return result
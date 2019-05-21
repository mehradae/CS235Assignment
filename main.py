import matplotlib.pylab as plt


def main():
    fileHandler()


def fileHandler():
    f = open("assignment_graph.txt", "r")

    # adjacency matrix - initialize with 0
    # see also: http://stackoverflow.com/questions/6667201/how-to-define-two-dimensional-array-in-python
    matrix = [[0 for i in range(21165)] for k in range(21165)]
    for i in f.readlines():
        str = i.splitlines()
        node = str[0].split(',')
        row = int(node[0]) - 1
        col = int(node[1]) - 1
        matrix[row][col] = 1
        plot(matrix)


def plot(matrix):
    plt.plt.spy(matrix)
    plt.show()


if __name__ == '__main__':
    main()

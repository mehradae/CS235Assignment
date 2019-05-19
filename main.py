import matplotlib.pylab as plt

def main():
    f = open("assignment_graph.txt", "r")
    graph = {}
    n = 0
    # for i in f.readline():
    #     node = i.split(',')
    #
    # print n1
    # # print n2

    # adjacency matrix - initialize with 0
    # see also: http://stackoverflow.com/questions/6667201/how-to-define-two-dimensional-array-in-python
    Matrix = [[0 for i in range(21165)] for k in range(21165)]
    for i in f.readlines():
        str = i.splitlines()
        node = str[0].split(',')
        row = int(node[0])-1
        col = int(node[1])-1
        Matrix[row][col] = 1
        # print("first= " + node[0] + ",second= " + node[1] + "\n")
    plt.plt.spy(Matrix)
    plt.show()

if __name__ == '__main__':
    main()

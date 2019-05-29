import matplotlib.pylab as plt


def main():
    matrixInit()


def matrixInit():
    f = open("assignment_graph.txt", "r")

    # adjacency matrix - initialize with 0
    # see also: http://stackoverflow.com/questions/6667201/how-to-define-two-dimensional-array-in-python
    matrix = [[0] * 21165 for _ in range(21165)]
    for i in f.readlines():
        temp = i.splitlines()
        node = temp[0].split(',')
        row = int(node[0]) - 1
        col = int(node[1]) - 1
        matrix[row][col] = 1
    spyPlot(matrix)
    degreeCount(matrix)


def spyPlot(matrix):
    plt.spy(matrix)
    plt.show()


def logPlot(degreeList):
    x = sorted(degreeList.keys())
    y = [degreeList[k] for k in x]
    plt.loglog(x, y, marker='*', linestyle='None')
    plt.xlim(1)
    plt.ylim(1)
    plt.xlabel('Degree')
    plt.ylabel('Node Count')
    plt.show()


def degreeCount(matrix):
    degreeList = {}
    for i in matrix:
        tmp = sum(i)
        if tmp in degreeList:
            degreeList[tmp] = degreeList.pop(tmp) + 1
        else:
            degreeList[tmp] = 1
    # summed_list = [sum(i) for i in matrix]
    # print(degreeList)
    logPlot(degreeList)


if __name__ == '__main__':
    main()

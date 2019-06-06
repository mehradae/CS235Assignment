import matplotlib.pylab as plt
import numpy as np
import scipy.sparse.linalg as spl
import scipy.spatial.distance as dist
from scipy import sparse


def main():
    # making the sparse matrix and add input file
    print('I\'m making the matrix from the imported file!')
    matrixInit()


def AbnormalGraph(matrix):
    counts = [sum(matrix[i]) for i in range(len(matrix[0]))]
    plt.plot(counts, marker='.', linestyle='None', markersize=1)
    plt.title('Abnormal Blocks')
    plt.xlabel('Nodes')
    plt.ylabel('Degree')
    plt.show()


def SVD(matrix):
    k = 80
    p_data = 0.0

    sp_matrix= sparse.csr_matrix(matrix)
    while p_data < 0.90:
        print("\nsolving svds for k =", k)
        U, S, V = spl.svds(sp_matrix, k=k)
        u = np.array(U)
        s = np.diag(S)
        v = np.array(V)
        m = (u @ s @ v)

        print(" SQE calc ...")
        sqe = dist.sqeuclidean(m.flatten(), sp_matrix.toarray().flatten())

        adj_tot = len(sp_matrix.nonzero()[0])

        p_data = 1 - (sqe / adj_tot)
        print(" Data reconstruction: ", p_data, "%")

        print("\nPlotting graphs ...")

        plt.subplot(121)
        plt.title("90% of Reconstructed Data")
        plt.spy(m, markersize=1, precision=0.1)
        plt.subplot(122)
        plt.title("Full Graph")
        plt.spy(matrix, markersize=1)
        plt.show()

        k += 1


def matrixInit():
    # adjacency matrix - initialize with 0
    # see also: http://stackoverflow.com/questions/6667201/how-to-define-two-dimensional-array-in-python
    print("creating adjacency matrix now ...")
    matrix = [[0.0] * 21165 for _ in range(21165)]
    print('Done !')
    with open("assignment_graph.txt") as file:
        for i in file:
            temp = i.splitlines()
            node = temp[0].split(',')
            row = int(node[0]) - 1
            col = int(node[1]) - 1
            matrix[row][col] = 1
    print('Now I plot your graphs!')
    # spyPlot(matrix)
    # degreeCount(matrix)
    # AbnormalGraph(matrix)
    SVD(matrix)
    print('Done !')


def spyPlot(matrix):
    plt.spy(matrix, marker='.', markersize=1)
    plt.show()


def logPlot(degreeList):
    x, y = zip(*(degreeList.items()))
    plt.loglog(x, y, marker='*', linestyle='None')
    plt.xlim(1)
    plt.ylim(1)
    plt.title('Degree Distribution')
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
    logPlot(degreeList)


if __name__ == '__main__':
    main()

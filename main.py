import matplotlib.pylab as plt
import numpy as np
import scipy.sparse.linalg as splin
import scipy.spatial.distance as dist
from scipy import sparse


def main():
    # making the sparse matrix and add input file
    print('I\'m making the matrix from the imported file!')
    # adjacency matrix - initialize with 0
    # see also: http://stackoverflow.com/questions/6667201/how-to-define-two-dimensional-array-in-python
    matrix = [[0] * 21165 for _ in range(21165)]

    with open("assignment_graph.txt") as file:
        for i in file:
            temp = i.splitlines()
            node = temp[0].split(',')
            row = int(node[0]) - 1
            col = int(node[1]) - 1
            matrix[row][col] = 1
    print('Done !')
    print("Now I start making the plots:")
    spyPlot(matrix)
    degreeCount(matrix)
    AbnormalGraph(matrix)
    SVD(matrix)
    print('All Done !')


def AbnormalGraph(matrix):
    print('\n\n Plotting abnormal graph')
    counts = [sum(matrix[i]) for i in range(len(matrix[0]))]
    plt.plot(counts, marker='.', linestyle='None', markersize=1)
    plt.title('Abnormal Blocks')
    plt.xlabel('Nodes')
    plt.ylabel('Degree')
    plt.show()
    print(' Done!')


def SVD(matrix):
    k = 80
    p_data = 0.0

    print('\n\n transport the data into sparse matrix.')
    sp_matrix = sparse.csr_matrix(matrix, dtype=float)
    while p_data < 0.90:
        print("  Computing SVD with k =", k)
        # making SVD matrices
        U, S, Vt = splin.svds(sp_matrix, k=k)
        print('  Done!')
        u = np.array(U)
        s = np.diag(S)
        v = np.array(Vt)
        # rebuilding the matrix
        Y = (u @ s @ v)

        print("  Euclidean square distance computing")
        distance = dist.sqeuclidean(Y.flatten(), sp_matrix.toarray().flatten())
        print('  Done!')
        total = len(sp_matrix.nonzero()[0])
        p_data = 1 - (distance / total)
        print("   Data reconstruction at ", p_data, "%")
        plt.subplot(1, 2, 1)
        plt.title("   90% of Reconstructed Data")
        plt.spy(Y, markersize=1, precision=0.1)
        plt.subplot(1, 2, 2)
        plt.title("Full Graph")
        plt.spy(matrix, markersize=1)
        plt.show()

        k += 1
    print('  Done!')
    print("\n  Computing SVD with k =5")
    U, S, Vt = splin.svds(sp_matrix, k=5)
    # plot top 5 result
    plt.title('Top 5 Left Singular Vector')
    plt.plot(U, markersize=1)
    plt.legend(('LSV: 1', 'LSV: 2', 'LSV: 3', 'LSV: 4', 'LSV: 5'), loc='best')
    plt.show()
    print("  Done!")

    print("\n  Computing SVD with k = 5")
    absU = np.absolute(np.array(U))

    indices = np.ones([100, 5], dtype=int)

    print("   finding top 100")
    for i in range(5):
        ind = np.argpartition(absU[:, i], -100)[-100:]
        indices[:, i] = ind[np.argsort(absU[:, i][ind])]

    TopU = np.zeros(U.shape)
    TopVt = np.zeros(Vt.shape)

    # Getting top 100
    for i in range(5):
        for j in range(indices.shape[0]):
            ind = indices[j, i]
            TopU[ind, i] = U[ind, i]
            TopVt[i, ind] = Vt[i, ind]
    print('   Done!')
    print("\n  Plot Top 100")
    plt.figure(figsize=(16, 10))
    for i in range(5):
        plt.subplot(2, 3, (i + 1))
        plt.title('SubGraph %i' % (i + 1))
        m = np.reshape(S[i] * TopU[:, i], (TopU.shape[0], 1)) * TopVt[i, :]
        plt.spy(m, markersize=1)

    plt.subplot(2, 3, 6)
    plt.title('All 5 in one Graph')
    DigS = np.diag(S)
    m_top = TopU @ DigS @ TopVt
    plt.spy(m_top, markersize=1)
    plt.show()
    print("  Done !")


def spyPlot(matrix):
    print(" Plotting the graph")
    plt.spy(matrix, marker='.', markersize=1)
    plt.show()
    print(' Done!')


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
    print('\n\n Plotting the degree count')
    degreeList = {}
    for i in matrix:
        tmp = sum(i)
        if tmp in degreeList:
            degreeList[tmp] = degreeList.pop(tmp) + 1
        else:
            degreeList[tmp] = 1
    logPlot(degreeList)
    print(' Done!')


if __name__ == '__main__':
    main()

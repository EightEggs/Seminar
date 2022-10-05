from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, DBSCAN
from matplotlib import pyplot as plt

moons, _ = make_moons(200, noise=.05, random_state=0)
n = len(moons)
x, y = moons[0:n, 0], moons[0:n, 1]
print(__name__)

if __name__ == '__main__':
    cluster_k = KMeans(2).fit_predict(moons)
    cluster_d = DBSCAN(eps=0.2).fit_predict(moons)
    plt.figure(0)
    plt.title('moons(before clustering)')
    plt.scatter(x, y)
    plt.figure(1)
    plt.title('moons(kmeans clustering)')
    plt.scatter(x, y, c=cluster_k)
    plt.figure(2)
    plt.title('moons(dbscan clustering)')
    plt.scatter(x, y, c=cluster_d)
    plt.show()

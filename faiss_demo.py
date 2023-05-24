import numpy as np
import faiss


def load_data():
    d = 64
    # 向量维度
    nb = 100000
    # 待索引向量size
    nq = 10000
    # 查询向量size
    np.random.seed(1234)
    # 随机种子确定
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.
    # 为了使随机产生的向量有较大区别进行人工调整向量
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.

    return d, xb, xq


def build_index(d, xb):
    index = faiss.IndexFlatL2(d)
    # 建立索引
    print(index.is_trained)
    # 输出true
    index.add(xb)
    # 索引中添加向量
    print(index.ntotal)
    # 输出100000

    return index


def search(index, xb, xq):
    k = 4
    # 返回每个查询向量的近邻个数
    D, I = index.search(xb[:5], k)
    # 检索check
    print(I)
    print(D)
    D, I = index.search(xq, k)
    # xq检索结果
    print(I[:5])
    # 前五个检索结果展示
    print(I[-5:])
    # 最后五个检索结果展示


def main():
    d, xb, xq = load_data()
    index = build_index(d, xb)
    search(index, xb, xq)


if __name__ == '__main__':
    main()

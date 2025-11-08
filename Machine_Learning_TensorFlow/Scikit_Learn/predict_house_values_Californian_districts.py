from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
    with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()
housing.info()
print(housing["ocean_proximity"].value_counts())
print(housing.describe())

housing.hist(bins=50, figsize=(20,15))
plt.show()





import numpy as np
#创建测试集(随机)  
def shuffle_and_split_data(data, test_ratio):
     shuffled_indices = np.random.permutation(len(data))
     test_set_size = int(len(data) * test_ratio)
     test_indices = shuffled_indices[:test_set_size]
     train_indices = shuffled_indices[test_set_size:]
     return data.iloc[train_indices], data.iloc[test_indices]

from sklearn.model_selection import train_test_split
#创建测试集(随机)
def test(test_size):
    train_set, test_set = train_test_split(housing, test_size=test_size, random_state=42)
    return train_set, test_set



from zlib import crc32

#创建测试集(基于标识符)始终保持一致  

def is_id_in_test_set(identifier, test_ratio):
     return crc32(np.int64(identifier)) < test_ratio * 2**32
"""identifier：某一行数据的唯一ID（比如用户ID、索引号）。
crc32(...)：对这个ID做CRC32哈希，得到一个0 ~ 2^32-1之间的整数。
test_ratio * 2**32：计算一个“阈值”，例如 test_ratio=0.2 → 阈值就是 20% 的 2^32。
return ... < ...：如果哈希值小于阈值，就把该ID分到测试集，否则分到训练集。"""

def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index()  # adds an `index` column
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")



from sklearn.model_selection import train_test_split
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
# 按 income_cat 分层抽样，划分 8:2
strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)
#stratify=housing["income_cat"]：告诉函数“按 income_cat 这列的分布来分层”。
#意思是：训练集和测试集中 income_cat 的比例要和原始数据一致。
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)



housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
plt.show()

housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))
plt.show()

                                                                                                                        


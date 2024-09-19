from functools import partial

from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.inne import INNE
from pyod.models.gmm import GMM
from pyod.models.kde import KDE
from pyod.models.lmdd import LMDD

from pyod.models.dif import DIF
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.sampling import Sampling
from pyod.models.kpca import KPCA
from pyod.models.lunar import LUNAR
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.auto_encoder import AutoEncoder


def get_dataset_names():
    mat_file_list = [
        "arrhythmia.mat",
        "cardio.mat",
        "glass.mat",
        "ionosphere.mat",
        "letter.mat",
        "lympho.mat",
        "mnist.mat",
        "musk.mat",
        "optdigits.mat",
        "pendigits.mat",
        "pima.mat",
        "satellite.mat",
        "satimage-2.mat",
        "shuttle.mat",
        "vertebral.mat",
        "vowels.mat",
        "wbc.mat",
    ]

    return mat_file_list


def get_models():
    model_list = [
        (ABOD, "ABOD"),
        (KNN, "KNN"),
        (partial(KNN, method='mean'), "Average KNN"),
        (partial(KNN, method='median'), "Median KNN"),
        (LOF, "LOF"),
        (IForest, "IForest"),
        (DIF, "Deep IForest"),
        (INNE, "INNE"),
        (MCD, "MCD"),
        (PCA, "PCA"),
        (KPCA, "KPCA"),
        (GMM, "GMM"),
        (LMDD, "LMDD"),
        (HBOS, "HBOS"),
        (COPOD, "COPOD"),
        (ECOD, "ECOD"),
        (KDE, "KDE"),
        (Sampling, "Sampling"),
        (LUNAR, "LUNAR"),
        (CBLOF, "CBLOF"),
        (OCSVM, "OCSVM"),
        (DeepSVDD, "DeepSVDD"),
        (AutoEncoder, "AutoEncoder"),
    ]
    return model_list

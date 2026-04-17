#make sure scikit-learn is pip installed
try:
    import sklearn
except ImportError as e:
    raise ImportError(
        "This module requires scikit-learn. Install it with:\n"
        "pip install scikit-learn"
    ) from e
  
from sklearn.neighbors import KernelDensity
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

import numpy as np
import matplotlib.pyplot as plt


def calc_test_value(class_1_vecs, class_2_vecs):
  mean_1=np.mean(class_1_vecs,axis=0)
  mean_2=np.mean(class_2_vecs,axis=0)

  observed_value=np.linalg.norm(mean_1-mean_2, ord=1)
  return observed_value

def do_a_permutatoin(class_1_vecs, class_2_vecs):
  all_vecs=np.concatenate((class_1_vecs, class_2_vecs))

  rng = np.random.default_rng()
  permuted_vecs = rng.permutation(all_vecs)

  

  new_class_1_vecs=permuted_vecs[:len(class_1_vecs)]
  new_class_2_vecs=permuted_vecs[len(class_1_vecs):]

 
  test_stat=calc_test_value(new_class_1_vecs, new_class_2_vecs)
  return test_stat

def do_a_permutation_test(class_1_vecs, class_2_vecs, num_permutations):
  observed_value=calc_test_value(class_1_vecs, class_2_vecs)
  
  test_stats=[None for _ in range(num_permutations)]
  for i in range(num_permutations):
    test_stat=do_a_permutatoin(class_1_vecs, class_2_vecs)
    test_stats[i]=test_stat

  p_value=np.sum(np.array(test_stats)>=observed_value)/num_permutations
  return p_value

def do_SVM_and_PCA(list1, list2, n_permutations=1000, cv_folds=10, random_state=42, name_1='class 0', name_2='class 1'):
  
    """
    Performs SVM classification between two sets of vectors,
    calculates a p-value via permutation test,
    and plots 3D PCA projection.

    Parameters:
    - list1, list2: lists or arrays of vectors (shape: n_samples x n_features)
    - n_permutations: number of permutations for p-value calculation (default=1000)
    - cv_folds: number of cross-validation folds (default=10)
    - random_state: random seed for reproducibility

    Returns:
    - accuracy: SVM cross-validated accuracy
    - p_value: permutation test p-value
    """

    np.random.seed(random_state)

    # Convert lists to arrays
    X1 = np.array(list1)
    X2 = np.array(list2)

    # Combine data and labels
    X = np.vstack([X1, X2])
    y = np.array([0]*len(X1) + [1]*len(X2))

    # Stratified cross-validation (preserves class balance)
    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state
    )

    # Define classifier
    clf = SVC(kernel='linear')

    # Compute real accuracy
    accuracy = np.mean(cross_val_score(clf, X, y, cv=cv))
    print(f"Observed accuracy: {accuracy:.4f}")

    # Permutation test
    p_value=do_a_permutation_test(X1,X2,n_permutations)
    print(f"Permutation p-value: {p_value:.4f}")

    # PCA for visualization
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        X_pca[y == 0, 0],
        X_pca[y == 0, 1],
        X_pca[y == 0, 2],
        label=name_1,
        s=50
    )

    ax.scatter(
        X_pca[y == 1, 0],
        X_pca[y == 1, 1],
        X_pca[y == 1, 2],
        label=name_2,
        s=50
    )

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(f'3D PCA Projection (SVM Acc={accuracy:.2f}, p={p_value:.3f})')
    ax.legend()

    plt.show()

    return accuracy, p_value

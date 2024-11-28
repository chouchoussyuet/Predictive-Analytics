import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def correct_labels_with_knn(X_train, y_train, X_test, y_pred):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    y_knn_pred = knn.predict(X_test)
    corrected_y_pred = []

    for i in range(len(y_pred)):
        neighbors_indices = knn.kneighbors([X_test[i]], n_neighbors=5, return_distance=False)[0]
        neighbor_labels = y_train.to_numpy()[neighbors_indices]
        neighbor_labels = neighbor_labels.astype(int).flatten()

        majority_label = np.bincount(neighbor_labels).argmax()

        if y_pred[i] != majority_label:
            corrected_y_pred.append(majority_label)
        else:
            corrected_y_pred.append(y_pred[i])

    return corrected_y_pred

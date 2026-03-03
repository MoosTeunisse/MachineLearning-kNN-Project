import numpy as np

class ManualKNNClassifier:
    """
    A manual (numpy only) implementation of KNN for classification.
    """
    def __init__(self, k=5, voting="uniform"):
        self.k = k
        self.voting = voting
    
    def fit(self, X, y):
        """
        Just store the training data, since KNN is a lazy learner.
        Changes made so far:
            - added integer labels for y
        """
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)

        self.classes_, self.y_train_int = np.unique(self.y_train, return_inverse=True)
        return self
    
    def predict(self, new_points, batch_size=256):
        """
        Return predictions for the test data (new_points).
        Changes made so far:
            - added batching
            - merged this with predict_class, which was needed for batching to work ya know (predict_class is commented below for reference)
        """
        X_test = np.asarray(new_points, dtype=float)
        X_train = self.X_train
        
        n_test = X_test.shape[0]
        predictions_int = []
        
        train_sq = np.sum(X_train * X_train, axis=1)
        
        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            Xbatch = X_test[start:end]
            
            batch_sq = np.sum(Xbatch * Xbatch, axis=1)
            
            dists = batch_sq[:, None] + train_sq[None, :] - 2 * Xbatch.dot(X_train.T)
            dists = np.maximum(dists, 0.0)
            
            k_eff = min(self.k, dists.shape[1])
            indices_k = np.argpartition(dists, k_eff - 1, axis=1)[:, :k_eff]
            
            for i in range(indices_k.shape[0]):
                nearest_indices = indices_k[i]
                nearest_labels_int = self.y_train_int[nearest_indices]
                
                if self.voting == "uniform":
                    most_occurring = np.bincount(nearest_labels_int, minlength=len(self.classes_))
                    predictions_int.append(int(np.argmax(most_occurring)))
                
                elif self.voting == "distance":
                    nearest_dists = np.sqrt(dists[i, nearest_indices])
                    nearest_weights = 1.0 / (nearest_dists + 1e-9)
                    
                    sums = np.bincount(nearest_labels_int, weights=nearest_weights, minlength=len(self.classes_))
                    predictions_int.append(int(np.argmax(sums)))
        
        return self.classes_[np.asarray(predictions_int)]
    
    
    
    # def predict_class(self, new_point):
    #     """
    #     Way to predict class of new point based on training data.
    #     Changes made so far:
    #         - added weighted voting (uniform and distance)
    #         - vectorized distance computation (this shit was way too fkn slow, still is tbh, batching is the next step)
    #     """
    #     #distances = [euclidean_distance(point, new_point) for point in self.X_train] (old way, very slowe)
    #     new_point = np.asarray(new_point, dtype=float)
        
    #     diff = self.X_train - new_point
    #     distances = np.sqrt(np.sum(diff ** 2, axis=1))
        
    #     #k_nearest_indices = np.argsort(distances)[:self.k] (old way, very slow)
    #     k_eff = min(self.k, distances.shape[0])
    #     k_nearest_indices = np.argpartition(distances, k_eff - 1)[:k_eff]
        
    #     if self.voting == "uniform":
    #         k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
    #         most_occurring = Counter(k_nearest_labels).most_common(1)[0][0]
    #         return most_occurring
        
    #     elif self.voting == "distance":
    #         k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
    #         k_nearest_weights = [1 / (d + 1e-9) for d in [distances[i] for i in k_nearest_indices]]
            
    #         label_weight_sum = {}
    #         for label, weight in zip(k_nearest_labels, k_nearest_weights):
    #             label_weight_sum[label] = label_weight_sum.get(label, 0) + weight
            
    #         most_weighted_label = max(label_weight_sum, key=label_weight_sum.get)
            
    #         return most_weighted_label


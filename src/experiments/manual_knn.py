import numpy as np

class ManualKNNClassifier:
    """
    A manual (numpy only) implementation of KNN for classification.
    """
    def __init__(self, k=5, voting="uniform", alpha=0.5, tie_break="nearest"):
        self.k = k
        self.voting = voting
        self.alpha = alpha
        self.tie_break = tie_break
    
    def fit(self, X, y):
        """
        Just store the training data, since KNN is a lazy learner.
        Changes made so far:
            - added integer labels for y
        """
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)

        self.classes_, self.y_train_int = np.unique(self.y_train, return_inverse=True)
        
        counts = np.bincount(self.y_train_int, minlength=len(self.classes_))
        self.class_prior_ = counts / counts.sum()
        
        return self
    
    def predict(self, new_points, batch_size=256):
        """
        Return predictions for the test data (new_points).
        Changes made so far:
            - added batching
            - merged this with predict_class, which was needed for batching to work ya know (predict_class is commented below for reference)
            - changed voting to make use of class priors, to account for the class imbalance in the dataset
        """
        X_test = np.asarray(new_points, dtype=float)
        X_train = self.X_train
        
        n_test = X_test.shape[0]
        predictions_int = []
        
        train_sq = np.sum(X_train * X_train, axis=1)
        
        num_ties = 0
        
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
                
                nearest_dists_sq = dists[i, nearest_indices]
                
                if self.voting == "uniform":
                    alpha = self.alpha
                    
                    votes = np.bincount(nearest_labels_int, minlength=len(self.classes_)).astype(float)
                    scores = votes / ((self.class_prior_ + 1e-12) ** alpha)
                    max_score = np.max(scores)
                    
                    tied = np.where(scores == max_score)[0]
                    if tied.size == 1:
                        predictions_int.append(int(tied[0]))
                    else:
                        num_ties += 1
                        pred = self._break_tie(nearest_labels_int, nearest_dists_sq, tied)
                        predictions_int.append(pred)
                
                elif self.voting == "distance":
                    alpha = self.alpha 
                    
                    nearest_dists = np.sqrt(dists[i, nearest_indices])
                    nearest_weights = 1.0 / (nearest_dists + 1e-9)

                    votes = np.bincount(nearest_labels_int, weights=nearest_weights, minlength=len(self.classes_)).astype(float)
                    scores = votes / ((self.class_prior_ + 1e-12) ** alpha)
                    max_score = np.max(scores)
                    
                    tied = np.where(np.isclose(scores, max_score, rtol=1e-9, atol=1e-9))[0]
                    if tied.size == 1:
                        predictions_int.append(int(tied[0]))
                    else:
                        num_ties += 1
                        pred = self._break_tie(nearest_labels_int, nearest_dists_sq, tied)
                        predictions_int.append(pred)
        print("ties encountered:", num_ties)
        
        return self.classes_[np.asarray(predictions_int)]
    
    def _break_tie(self, nearest_labels_int, nearest_dists_sq, tied_classes):
        """
        Break ties between classes in tied_classes.
        tie_break="nearest": pick the class of the closest neighbor among tied classes.
        """
        if self.tie_break == "nearest":
            mask = np.isin(nearest_labels_int, tied_classes)
            j = int(np.argmin(nearest_dists_sq[mask]))
            return int(nearest_labels_int[mask][j])

        elif self.tie_break == "min_class":
            return int(np.min(tied_classes))
    
    
    #----------------------------------------OLD---------------------------------------------
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


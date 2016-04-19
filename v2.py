class MyClassifierTwo(MyClassifier):

    def __init__(self):
        self.model = []
        self.n_classes = 0
        self.n_features = 0
        self.used_features = set()
        self.weights = []
        self.curr_weight = 1.0
        self.weight_change = 0.5

    def fit(self, X, Y):
        self.n_features = X.shape[1]
        self.n_classes = len(set(Y))
        for i in range(5):
            best_split, curr_least_impurity = self.fit_once(X, Y)
            print "%d new tree(s) found, driving impurity measure down to %f" % (len(best_split), curr_least_impurity)
            print best_split
            self.used_features = self.used_features.union([e[0] for e in best_split])
            self.model += best_split
            self.weights += [self.curr_weight] * len(best_split)
            self.curr_weight = self.curr_weight * self.weight_change
        return self

    def predict(self, X):
        res = []
        for obs in X:
            prob = numpy.array([0.0] * self.n_classes)
            ind = 0
            for estimator in self.model:
                feat_idx, feat_val, cnt_l, cnt_r = estimator
                weight = self.weights[ind]
                if obs[feat_idx] <= feat_val:
                    prob += numpy.array([weight * c / float(sum(cnt_l)) for c in cnt_l])
                else:
                    prob += numpy.array([weight * c / float(sum(cnt_r)) for c in cnt_r])
            res.append(numpy.argmax(prob))
        return res

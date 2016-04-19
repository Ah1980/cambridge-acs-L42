import numpy

class MyClassifier:
    def __init__(self):
        self.model = []
        self.n_classes = 0
        self.n_features = 0
        self.used_features = set()

    def compute_gini(self, cnt):
        return 1.0 - sum([c*c for c in cnt]) / float(sum(cnt) * sum(cnt))

    def fit_once(self, X, Y):
        n = X.shape[0]
        p = X.shape[1]
        k = len(set(Y))
        best_split = None
        cnt = [0] * k
        for y in Y:
            cnt[y] += 1
        impurity = self.compute_gini(cnt)
        curr_least_impurity = impurity
        for feat_idx in range(p):
            if feat_idx in self.used_features:
                continue
            all_feat = [obs[feat_idx] for obs in X]
            for feat_val in all_feat:
                cnt_l = [0] * k
                cnt_r = [0] * k
                for i in range(n):
                    if all_feat[i] <= feat_val:
                        cnt_l[Y[i]] += 1
                    else:
                        cnt_r[Y[i]] += 1

                if sum(cnt_l) == 0 or sum(cnt_r) == 0:
                    continue

                impurity_l = self.compute_gini(cnt_l)
                impurity_r = self.compute_gini(cnt_r)
                avg_impurity = float(sum(cnt_l)) / n * impurity_l + float(sum(cnt_r)) / n * impurity_r
                if avg_impurity < curr_least_impurity:
                    best_split = [(feat_idx, feat_val, cnt_l, cnt_r)]
                    curr_least_impurity = avg_impurity
                elif avg_impurity == curr_least_impurity:
                    best_split.append((feat_idx, feat_val, cnt_l, cnt_r))
        return (best_split, curr_least_impurity)

    def fit(self, X, Y):
        self.n_features = X.shape[1]
        self.n_classes = len(set(Y))
        for i in range(5):
            best_split, curr_least_impurity = self.fit_once(X, Y)
            print "%d new tree(s) found, driving impurity measure down to %f" % (len(best_split), curr_least_impurity)
            print best_split
            self.used_features = self.used_features.union([e[0] for e in best_split])
            self.model += best_split
        return self

    def predict(self, X):
        res = []
        for obs in X:
            prob = numpy.array([0.0] * self.n_classes)
            for estimator in self.model:
                feat_idx, feat_val, cnt_l, cnt_r = estimator
                if obs[feat_idx] <= feat_val:
                    prob += numpy.array([c / float(sum(cnt_l)) for c in cnt_l])
                    # print prob
                else:
                    prob += numpy.array([c / float(sum(cnt_r)) for c in cnt_r])
            res.append(numpy.argmax(prob))
        return res

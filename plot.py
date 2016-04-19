backup = clfr.model

res = []
for num_tree in range(1, len(backup)):
    clfr.model = backup[0:num_tree]
    X_test = exp_all[testing_idx]
    Yprime_test = clfr.predict(X_test)
    res.append((num_tree, sum(health_classes[testing_idx] == Yprime_test)/float(len(testing_idx))))


# This is necessary for using matplotlib in jupyter
get_ipython().magic(u'matplotlib inline')

# Import the necessary modules and libraries
import numpy as np
import matplotlib.pyplot as plt

Xpts = [r[0] for r in res]
y = [r[1] for r in res]

# Plot the results
plt.figure()
plt.scatter(Xpts, y, c="k", label="Accuracy")
plt.xlabel("Number of trees")
plt.ylabel("Accuracy")
plt.title("Decision Forests Accuracy")
plt.vlines(3, 0.1, 0.8)
plt.vlines(25, 0.1, 0.8)
plt.vlines(55, 0.1, 0.8)
plt.vlines(79, 0.1, 0.8)
plt.vlines(97, 0.1, 0.8)
plt.legend()
plt.show()

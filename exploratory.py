import numpy

from sklearn import tree
X = exp_all[training_idx]
Y = health_classes[training_idx]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)


X_test = exp_all[testing_idx]
Yprime_test = clf.predict(X_test)


print collections.Counter(health_classes[testing_idx] == Yprime_test)
print collections.Counter(zip(health_classes[testing_idx], health_classes[testing_idx] == Yprime_test))


# ## AdaBoost

X = exp_all[training_idx]
Y = health_classes[training_idx]


from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier()
clf = clf.fit(X, Y)
X_test = exp_all[testing_idx]
Yprime_test = clf.predict(X_test)


print collections.Counter(health_classes[testing_idx] == Yprime_test)
print collections.Counter(zip(health_classes[testing_idx], health_classes[testing_idx] == Yprime_test))

print collections.Counter(health_classes[testing_idx] == Yprime_test)
print collections.Counter(zip(health_classes[testing_idx], health_classes[testing_idx] == Yprime_test))


# In[180]:

clf.decision_function(X_test[0:10])


# In[213]:

import operator
imp = zip(range(0, 22283), clf.feature_importances_)
imp.sort(key=operator.itemgetter(1), reverse=True)
imp[0:20]


# In[182]:

health_classes[testing_idx][0:10]


# In[168]:

print collections.Counter(health_classes[testing_idx] == Yprime_test)
print collections.Counter(zip(health_classes[testing_idx], health_classes[testing_idx] == Yprime_test))


# #### Is AdaBoost deterministic?

# ## Bagging

# In[532]:

from sklearn.ensemble import BaggingClassifier
clf = BaggingClassifier()
clf = clf.fit(X, Y)
X_test = exp_all[testing_idx]
Yprime_test = clf.predict(X_test)


# In[533]:

print collections.Counter(health_classes[testing_idx] == Yprime_test)
print collections.Counter(zip(health_classes[testing_idx], health_classes[testing_idx] == Yprime_test))


# #### Bagging with more estimators

# In[105]:

from sklearn.ensemble import BaggingClassifier
clf = BaggingClassifier(n_estimators=50)
clf = clf.fit(X, Y)
X_test = exp1[testing_idx]
Yprime_test = clf.predict(X_test)


# In[106]:

print collections.Counter(health_classes[testing_idx] == Yprime_test)
print collections.Counter(zip(health_classes[testing_idx], health_classes[testing_idx] == Yprime_test))


# ##### Another run may give very different results

# In[108]:

print collections.Counter(health_classes[testing_idx] == Yprime_test)
print collections.Counter(zip(health_classes[testing_idx], health_classes[testing_idx] == Yprime_test))


# #### Bagging with even more estimators

# In[130]:

from sklearn.ensemble import BaggingClassifier
clf = BaggingClassifier(n_estimators=150)
clf = clf.fit(X, Y)
X_test = exp1[testing_idx]
Yprime_test = clf.predict(X_test)


# In[131]:

print collections.Counter(health_classes[testing_idx] == Yprime_test)
print collections.Counter(zip(health_classes[testing_idx], health_classes[testing_idx] == Yprime_test))


# ### Gradient Boosting

# In[534]:

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
clf = clf.fit(X, Y)
X_test = exp_all[testing_idx]
Yprime_test = clf.predict(X_test)


# In[535]:

print collections.Counter(health_classes[testing_idx] == Yprime_test)
print collections.Counter(zip(health_classes[testing_idx], health_classes[testing_idx] == Yprime_test))


# In[121]:

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(max_depth=5)
clf = clf.fit(X, Y)
X_test = exp1[testing_idx]
Yprime_test = clf.predict(X_test)


# In[123]:

print collections.Counter(health_classes[testing_idx] == Yprime_test)
print collections.Counter(zip(health_classes[testing_idx], health_classes[testing_idx] == Yprime_test))


# In[ ]:




# ### RandomForestClassifier

# In[536]:

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf = clf.fit(X, Y)
X_test = exp_all[testing_idx]
Yprime_test = clf.predict(X_test)


# In[537]:

print collections.Counter(health_classes[testing_idx] == Yprime_test)
print collections.Counter(zip(health_classes[testing_idx], health_classes[testing_idx] == Yprime_test))


# In[127]:

print collections.Counter(health_classes[testing_idx] == Yprime_test)
print collections.Counter(zip(health_classes[testing_idx], health_classes[testing_idx] == Yprime_test))


# #### adding more trees do not help much

# In[128]:

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=150)
clf = clf.fit(X, Y)
X_test = exp1[testing_idx]
Yprime_test = clf.predict(X_test)


# In[129]:

print collections.Counter(health_classes[testing_idx] == Yprime_test)
print collections.Counter(zip(health_classes[testing_idx], health_classes[testing_idx] == Yprime_test))


# In[15]:

# This is necessary for using matplotlib in jupyter
get_ipython().magic(u'matplotlib inline')

# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(X, y, c="k", label="data")
plt.plot(X_test, y_1, c="g", label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, c="r", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()

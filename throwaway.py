from util import Counter
features = Counter()
features["a"] = 1
features["b"] = 2

weights = {"a":-7, "b": 20}

print(features*weights)
# K-Means Clustering From Scratch

This project implements K-Means using only NumPy.  
Files included:
- `kmeans.py` — implementation
- `README.md` — explanation

Run example:
```python
import numpy as np
from kmeans import kmeans

X = np.random.randn(300, 2)
centroids, labels = kmeans(X, k=3)
```

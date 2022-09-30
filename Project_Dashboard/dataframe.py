import os
import pandas as pd
import numpy as np
path = "apple"
new = pd.read_csv(os.path.join(path, "labelled_reviews.csv"))
reviews = np.array(new["Review Text"])
for i in range(len(reviews)):
    if new["Score"].iloc[i] >= 3:
        new["Class"].iloc[i] = 'Positive'
    else:
        new["Class"].iloc[i] = 'Negative'

new.to_csv("labelled_reviews.csv")



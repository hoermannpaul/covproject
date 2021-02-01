'''
Apply model on test images. Model is initialized with weights from training
Afterwards Jaccard distance is calculated
'''

#%%
from main import get_data
from unet import UNet
from pathlib import Path
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import jaccard_score
#%%

test_images = get_data("data/images/test")
test_masks = get_data("data/masks/test")
size = 256, 144
model = UNet(size)
checkpoint_dir = Path().cwd() / "weights" / "run1"
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)

#%%

#predictions = model.predict(test_images)
#pickle.dump(predictions, open("src/predictions.pickle", "wb"))
predictions = pickle.load(open("src/predictions.pickle", "rb"))
ground_truths = test_masks

#%%

print("precictions: ", predictions.shape)
print("ground_truths: ",ground_truths.shape)

#%%

predictions_flat = predictions.flatten()
ground_truths_flat = ground_truths.flatten()

#%%
# plt.figure()
# plt.hist(predictions_flat, bins = 50)
# plt.figure()
# plt.hist(ground_truths_flat)
# plt.show()

# -----------------------------------------------------------------------------------
jaccards = []
jaccard_norm = 0
from tqdm import tqdm
for threshold in tqdm(range(0,256,1)):    
    
    jaccard = 0    
    for prediction, ground_truth in zip(predictions, ground_truths):        
        prediction = prediction.flatten()
        
        # normalize [0:255]   
        prediction -= np.min(prediction)
        prediction /= np.max(prediction)
        prediction *= 255.0
        prediction = prediction.astype(np.int32)

        prediction[prediction < threshold] = 0
        prediction[prediction >= threshold] = 255

        ground_truth = ground_truth.flatten()
        # ground_truth[ground_truth < threshold] = 0
        # ground_truth[ground_truth >= threshold] = 1
        # ground_truth = ground_truth.astype(np.int32)

        #score = jaccard_score(ground_truth, prediction)
        score = jaccard_similarity_score(ground_truth, prediction) # deprecated
        jaccard += score
        
    jaccard = jaccard / predictions.shape[0]    
    jaccards.append(jaccard)

pickle.dump(jaccards, open("src/jaccard.pickle", "wb"))
print("DONE")


# %%
# -----------------------------------------------------------------------------------
import pickle
import matplotlib.pyplot as plt
import numpy as np
jac = pickle.load(open("src/jaccard.pickle", "rb"))
jac_max = np.array(jac).max()
plt.plot(range(0,256,1),jac)
plt.xlabel("threshold")
plt.ylabel("jaccard")
plt.hlines(jac_max, 0, 256, linestyles="dashed")
plt.text(0, 0.9, round(jac_max,2))
plt.title("Jaccard similarity score")
plt.grid()
plt.show()

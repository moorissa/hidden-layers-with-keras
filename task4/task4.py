import os
import numpy as np
from keras.preprocessing import image
from keras import applications
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

images = [f for f in os.listdir("/rigel/edu/coms4995/datasets/pets")]
with open("/rigel/edu/coms4995/users/jwh2163/homework-v-moorissa/list.txt", "r") as f:
    check = f.readlines()

# store all images

images_list = []
cluster_ids = []
for line in check:
    if line[0] == "#":
        pass
    else:
        line = line.strip()
        line = line.split()
        image_name = line[0] + ".jpg"
        cluster_id = line[1]
        images_list.append(image.load_img("/rigel/edu/coms4995/datasets/pets/"+image_name, target_size=(224, 224))) 
        cluster_ids.append(cluster_id)
        
X = np.array([image.img_to_array(img) for img in images_list])

model = applications.VGG16(include_top=False, weights='imagenet')
X_pre = preprocess_input(X)
features = model.predict(X_pre)
features_ = features.reshape(7349, -1)

X_train, X_test, y_train, y_test = train_test_split(features_, cluster_ids, stratify=cluster_ids)

lr = LogisticRegression().fit(X_train, y_train)
print('Train score:',lr.score(X_train, y_train))
print('Test score:',lr.score(X_test, y_test))

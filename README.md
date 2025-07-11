# tools to train yolo and compare detected objects

here is my code for detecting same objects on multiple independent images
some my own notebooks and python tools to prepare datasets to train yolo.
datasets prepaired with cvat, so all code is for

```src/pipline_final.py``` is main file, all starts from here
main pipline points:
- load images
- format images
- detect objects on images with yolo
- cut object from image
- calc cutted object embedding (I used ```facebook/dinov2-large``` as embedding feature exxtraction model)
- compare embeddings with some treshold (euclidian similarity)
- verify similar object pairs (i used opencv-sift + skimage-canny)

or shorter:
- detect
- get embedding
- search similars
- verify similars

Pipline creates a lot of not required files, but it is cumfortable to run every single step independently

Also I tried to train pretrained vae to get embeddings, but this was bad idea in my case (may be I am wrong and it is usefull).

And I tried to use several verification methods, but in my case object was too similar. Ckeck some compare methods using ```src/image_comparator.py``` + ```src/test_comparing_methods.py```.

And I tried to train some small NN to translate embeddings to anouther tensor which is more compatible for euclidian comparing (```src/similarity_model_learning```). But i has too small dataset and it seems like my model overfitting or i make something wrong.
If it is interesting, first I run pipline of may data without this small NN, then check answer myself ot get dataset (just deleting not same object pairs from folder with debug data manualy). I think in my case main problem was in classes imbalanse (2 classes: same, not same; not same objects embeddings was about 95% of all pairs)

All settings is in ```src/settings_and_utils.py```

On my opinion possible way to upgrade objects comparing is to use https://github.com/verlab/accelerated_features as verification method

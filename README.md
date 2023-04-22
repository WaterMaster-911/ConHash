





# **Supervised Conformer Hashing with Entropy-Balanced Loss for Large-Scale Image Retrieval**(ConHash)

![Pipeline](.\pic\Pipeline.png)

# **Requirements**

install PyTorch 1.7.0+ and torchvision 0.8.1+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):

```
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2
pip install torchvision==0.5.0
```

## Dataset

There are three different configurations for cifar10

- config["dataset"]="cifar10" will use 1000 images (100 images per class) as the query set, 5000 images( 500 images per class) as training set , the remaining 54,000 images are used as database.
- config["dataset"]="cifar10-1" will use 1000 images (100 images per class) as the query set, the remaining 59,000 images are used as database, 5000 images( 500 images per class) are randomly sampled from the database as training set.
- config["dataset"]="cifar10-2" will use 10000 images (1000 images per class) as the query set, 50000 images( 5000 images per class) as training set and database.

You can download ImageNet [here](https://github.com/thuml/HashNet/tree/master/pytorch) where is the data split copy from,Place the imageNet dataset under path `./dataset/imagenet`

```
./dataset/imagenet
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```



## How to Run

Download the Conformer pretrained models from official repository and keep under pretrained Conformer directory:

**[Conformer-B](https://drive.google.com/file/d/1oeQ9LSOGKEUaYGu7WTlUGl3KDsQIi0MA/view)**  **[Conformer-S](https://drive.google.com/file/d/1mpOlbLaVxOfEwV4-ha78j_1Ebqzj2B83/view)**  **[Conformer-Ti](https://drive.google.com/file/d/19SxGhKcWOR5oQSxNUWUM2MGYiaWMrF1z/view)** 

Place the pretrained models under path `./preload`

You can easily train just by

```
python main.py  
```




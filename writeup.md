# **Traffic Sign Recognition** 

---

## Reflections: 

### Data Preprocessing

The biggest challenge (the test accuracy needs to be at least 0.89 and the validation set accuracy needs to be at least 0.93.) didn’t come from the CNN architecture designs and tuning but the data preprocessing. I used grayscale conversation, image data normalization, and histogram equalization. 

* For grayscale conversation, I immediately used it after realizing the color may not an important factor and good results gained from the first project. I didn’t have the opportunity to compare its effectiveness since I had the grayscale preprocessing at the beginning. 

* For image data normalization, I followed the project instruction: “For image data, (pixel - 128)/ 128 is a quick way to approximately normalize the data and can be used in this project“, the image is normalized between [-1, 1] because the image range [0, 255]. I kept trying to improve the test accuracy for days by improving CNN settings but never considered to change the normalization range until I noticed “Minimally, the image data should be normalized so that the data has mean zero and equal variance”, which made me feel the normalization can play a big role to affect the training effect. Instead of [-1, 1], I used (b - a) * ( (img - 0) / (255 - 0) ) + a with a = 0.1 and b = 0.9 to normalize the image data between [0.1, 0.9]. The test actuary immediately changed from 0.89 to 0.96. 

* For histogram equalization, due to the provided training data class varies as shown in the data histogram plot, we can generate additional more training images by rotating images by small angles and make each class has almost the same minimum number of training images. This preprocessing was proven to be optional for this project because the test accuracy improvement is negligible (from 0.961 to 0.962). I guess the reason is by simply generating additional training images by small rotation can’t provide extra useful features for the NN. 

Summary: “Minimally, the image data should be normalized so that the data has mean zero and equal variance. Other pre-processing steps are optional. You can try different techniques to see if it improves performance.” Listen to the project instructions SERIOUSLY! 

### CNN Designs 

I used three CNNs to conduct the training: LeNet, LeNet1 and LeNet2. The LeNet and LeNet1 are the [LeNet-5](https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb), and LeNet2 used the [improved LeNet](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) architecture. The LeNet has exactly the same architecture as the [LeNet-5](https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb), and LeNet used the same architecture with deeper filter size. 

For LeNet3, as stated in the [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf): "Usual ConvNets ([LeNet-5](https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb)) are organized in strict feed-forward layered architectures in which the output of one layer is fed only to the layer above." Contrary to the traditional LeNet, only the output of the second stage is fed to the classifier, in the [improved LeNet](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), the output of the first stage is also branched out and fed to the classifier. By adding the procedure, the training yielded higher accuracies than the traditional method.

Here is the training results: 

* LeNet: rate = 0.001, EPOCHS = 100, BATCH_SIZE = 128

  The training validation accuracy is 0.937, the testing accuracy is 0.922.
  
* LeNet1: rate = 0.001, EPOCHS = 40, BATCH_SIZE = 128

  The training validation accuracy is 0.970, the test accuracy is 0.963.
 
* LeNet2, rate = 0.001, EPOCHS = 100, BATCH_SIZE = 128

  The training validation accuracy is 0.962, the test accuracy is 0.950. 

Summary: The improved (LeNet2) CNN has a better training result compared to the traditional LeNet (LeNet) using similar layer settings. In the same CNN architecture, deeper filter size results in better training result but more computational power.  

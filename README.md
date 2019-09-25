## Project: Build a Traffic Sign Classifer 

### Install TensorFlow with GPU 

In the second project of "Self-Driving Car" Nanoprogram program, we are required to use convolutional neural networks (LeNet) to classify traffic signs. In order to gain a quicker training process, TensorFlow with GPU support need to be installed properly first. 

Besides the [official web instruction](https://www.tensorflow.org/install/), the following step by step resources are helpful:
* [Install Tensorflow (GPU version) for Windows and Anaconda](https://www.youtube.com/watch?v=Ebo8BklTtmc&t=673s)
* [Tensorflow GPU fails to find CUDA](https://github.com/tensorflow/tensorflow/issues/5968)
* [How to fix "python is not recognized as an internal or external command"](https://www.youtube.com/watch?v=uXqTw5eO0Mw)

You can do the test if the Tensorflow take the GPU advantage using the following method: [How to tell if Jupyter notebook is using GPU](https://discussions.udacity.com/t/how-to-tell-if-jupyter-notebook-is-using-gpu/217660)

There is one big problem if you installing tensorflow following exactly the [official web instruction](https://www.tensorflow.org/install/). When you don't finish the project noninterruptedly, you close the Anaconda and open it later, the following error message will show and the CMD will refuse to work: 

"
usage: conda [-h]
{keygen,sign,unsign,verify,unpack,install,install-scripts,convert,version,help}
...
conda: error: invalid choice: '..checkenv' (choose from 'keygen', 'sign', 'unsign', 'verify', 'unpack', 'install', 'install-scripts', 'convert', 'version', 'help')
"

I searched online and realized the issue is still very fresh, the problem is still being discussed in [conda command failure #6171](https://github.com/ContinuumIO/anaconda-issues/issues/6171). I finally solved my problem following [this](https://stackoverflow.com/a/46493533/8936445). In short, after activating the created environment, instead of using "pip install --ignore-installed --upgrade tensorflow-gpu", you HAVE TO use "pip install tensorflow-gpu" to avoid above issue. More detailed description can be found [here](https://wangjieleo.wixsite.com/jwang/single-post/2017/11/08/Installing-TensorFlow-with-GPU-support). 

---
### Data Preprocessing

The biggest challenge (the test accuracy needs to be at least 0.89 and the validation set accuracy needs to be at least 0.93.) didn’t come from the CNN architecture designs and tuning but the data preprocessing. I used grayscale conversation, image data normalization, and histogram equalization. 

* For grayscale conversation, I immediately used it after realizing the color may not an important factor and good results gained from the first project. I didn’t have the opportunity to compare its effectiveness since I had the grayscale preprocessing at the beginning. 

* For image data normalization, I followed the project instruction: “For image data, (pixel - 128)/ 128 is a quick way to approximately normalize the data and can be used in this project“, the image is normalized between [-1, 1] because the image range [0, 255]. I kept trying to improve the test accuracy for days by improving CNN settings but never considered to change the normalization range until I noticed “Minimally, the image data should be normalized so that the data has mean zero and equal variance”, which made me feel the normalization can play a big role to affect the training effect. Instead of [-1, 1], I used (b - a) * ( (img - 0) / (255 - 0) ) + a with a = 0.1 and b = 0.9 to normalize the image data between [0.1, 0.9]. The test actuary immediately changed from 0.89 to 0.96. 

* For histogram equalization, due to the provided training data class varies as shown in the data histogram plot, we can generate additional more training images by rotating images by small angles and make each class has almost the same minimum number of training images. This preprocessing was proven to be optional for this project because the test accuracy improvement is negligible (from 0.961 to 0.962). I guess the reason is by simply generating additional training images by small rotation can’t provide extra useful features for the NN. 

Summary: “Minimally, the image data should be normalized so that the data has mean zero and equal variance. Other pre-processing steps are optional. You can try different techniques to see if it improves performance.” Listen to the project instructions SERIOUSLY! 

---

### CNN Designs 

I used three CNNs to conduct the training: LeNet, LeNet1 and LeNet2. The LeNet and LeNet1 are the [LeNet-5](https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb), and LeNet2 used the [improved LeNet](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) architecture. The LeNet has exactly the same architecture as the [LeNet-5](https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb), and LeNet used the same architecture with deeper filter size. 

For LeNet3, as stated in the [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf): "Usual ConvNets ([LeNet-5](https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb)) are organized in strict feed-forward layered architectures in which the output of one layer is fed only to the layer above." Contrary to the traditional LeNet, only the output of the second stage is fed to the classifier, in the [improved LeNet](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), the output of the first stage is also branched out and fed to the classifier. By adding the procedure, the training yielded higher accuracies than the traditional method.

The input of all the tested networks is 32x32x1 image and the output is the probability of the 43 possible traffic signs.
 
The final model used for training is LeNet1 with the following layers:

| Layer         		|     Description	        					| Input |Output| 
|:---------------------:|:---------------------------------------------:| :----:|:-----:|
| Convolution 5x5     	| 1x1 stride, valid padding, ReLU activation 	|32x32x1 |28x28x32|
| Max pooling			| 2x2 stride				        		        |28x28x32|14x14x32|
| Convolution 5x5 	    | 1x1 stride, valid padding, ReLU activation 	|14x14x32|10x10x96|
| Max pooling			| 2x2 stride              	   					|10x10x96|5x5x96|
| Flatten				| 3 dimensions -> 1 dimension					|5x5x96| 2400|
| Fully Connected | ReLU activation, Dropout with keep_prob=0.5 to prevent overfitting 	|2400|600|
| Fully Connected | ReLU activation, Dropout with keep_prob=0.5 to prevent overfitting 	|600|150|
| Fully Connected | output = number of traffic signs   	|150| 43|


Here is the training results: 

* LeNet: rate = 0.001, EPOCHS = 100, BATCH_SIZE = 128

  The training validation accuracy is 0.937, the testing accuracy is 0.922.
  
* LeNet1: rate = 0.001, EPOCHS = 40, BATCH_SIZE = 128

  The training validation accuracy is 0.970, the test accuracy is 0.963.
 
* LeNet2, rate = 0.001, EPOCHS = 100, BATCH_SIZE = 128

  The training validation accuracy is 0.962, the test accuracy is 0.950. 

Summary: The improved (LeNet2) CNN has a better training result compared to the traditional LeNet (LeNet) using similar layer settings. In the same CNN architecture, deeper filter size results in better training result but more computational power.  

---

### Test on New Images

I downloaded six traffic sign images online to test the trained NN. Even all of them fall in the training data category, the trained NN never "see" them before. 

| Image			        |     Prediction		| 
|:---------------------:|:---------------------:| 
| Dangerous curve to the left  | Dangerous curve to the left  | 
| Go straight or left  		| Go straight or left 	|
| Priority road			| Priority road					|
| Right-of-way at the next intersection		| Right-of-way at the next intersection					|
| Wild animals crossing		| Wild animals crossing  |
| Yield | Yield |

6 of 6 correct = 100% 

Compared with trained LeNet1 test accuracy 0.963, the new images test accuracy showed no surprise. This is due to the fact the new tested images are all in the training category and I had a well trained NN. 

In the softmax probabilities, it shows the NN is very confident with its prediction (100%) with no prediction false. This may because the newly tested traffic signs are different enough to the second and third guess. I didn't test the signs isn't in the training set, because it's meaningless for the trained NN to give any predictive result. No matter how confident the predictive results are, they are all wrong. 

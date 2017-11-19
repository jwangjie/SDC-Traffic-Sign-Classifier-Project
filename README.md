## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Install TensorFlow with GPU support
---
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


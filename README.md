# lstm-classification

### Requirements
* GDAL -- [Homepage](http://www.gdal.org)
* TensorFlow -- [Homepage](https://www.tensorflow.org)

### Install Gdal
Get gdal development libraries:
```shell
$ sudo apt-add-repository ppa:ubuntugis/ubuntugis-unstable
$ sudo apt-get update
$ sudo apt-get install libgdal-dev
$ sudo apt-get install python3-dev
$ sudo apt-get install gdal-bin python3-gdal
```

### Create and activate a virtual environment
```shell
$ virtualenv env -p python3
$ source env/bin/activate
```
### Install Numpy
```shell
(env) $ pip3 install numpy
```
### Install GDAL
```shell
(env) $ pip3 install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal"
```
Install Others Requirements
```shell
(env) $ pip3 install -r requirements.txt
```

### Install Tensorflow with GPU
```
Install tensorflow-gpu
```shell
(env) $ pip3 install tensorflow-gpu==2.0.0b1
```

### 

nvidia-smi 

dpkg -l | grep cudnn

https://stackabuse.com/solving-sequence-problems-with-lstm-in-keras/

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
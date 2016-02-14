# netron
Smart and distributed (hyper)parameter search for Neural nets and other models.

## !! WARNING !!
If you are going to play with creating your own clusters it is highly encouraged
to use a separate AWS account (make sure to provide proper path to a credentials file).  
Currently when cluster is terminated it kills all running instances it can find.  
Isolating netron clusters from other running intances is in TODO.  

## TODO
1. Isolate netron clusters from other instances.  
2. JobRunner for a cluster.  
3. JobReporter. How to store the progress of training? How to display the progress? Separate dashboard?  
4. Plugins for model evolution:  
    4.1 Simple GridSearch  
    4.2 NEAT  
    4.3 QLearning?  
    4.4 ...  

## Installation
1. Install python dependencies: `sudo pip install -r requirements.txt`  
2. Install latest [VirtualBox](https://www.virtualbox.org/wiki/Downloads) and [vagrant](https://www.vagrantup.com/downloads.html).  
3. Start a vagrant box: `vagrant up`. It will start up a virtual machine with a MongoDB running inside of a docker container.

## How to run
1. Start a server: `python netron/server/JobHTTPServer.py`  
2. Start a worker: `python netron/worker/Worker.py`  

## Creating training data for workers
Training data must be stored as a compressed numpy file in
`server/static/data/`. To create such file for your training data:  
```python
np.savez_compressed("data_filename.npz", x_train = your_x_train, y_train = your_y_train)
```

Where `your_x_train` and `your_y_train` are numpy arrays.

## Contribution
1. Open a separate branch and make your changes there.  
2. Open a pull request to master (For now. It'll change later).


# netron
Smart and distributed hyperparameter/architecture search for Neural nets and other models.

## !! WARNING !!
If you are going to play with creating your own clusters it is highly encouraged
to use a separate AWS account (make sure to provide proper path to a credentials file).  
Currently when cluster is terminated it kills all running instances it can find.  
Isolating netron clusters from other running intances is in TODO.  

## TODO
1. Isolate netron clusters from other instances.  
2. JobReporter. How to store the progress of training? How to display the progress? Separate dashboard?  
3. Plugins for model evolution:  
    * GridSearch  
    * RandomSearch  
    * NEAT/HyperNEAT  
    * Bayesian Optimization  
    * Reversible learning  
    * QLearning?  
    * ...  

## Installation
1. Install python dependencies: `sudo pip install -r requirements.txt`  
2. Install latest [VirtualBox](https://www.virtualbox.org/wiki/Downloads) and [vagrant](https://www.vagrantup.com/downloads.html).  
3. Start a vagrant box: `vagrant up`. It will start up a virtual machine with a MongoDB running inside of a docker container.  

If you are running on a Linux server, you don't need vagrant and Virtual Machine and can simply start Mongo in a docker container:  
1. `docker pull mongo`  
2. `docker run -p 27017:27017 --name netron-mongo -d mongo`  
And then to stop docker container `docker stop netron-mongo` and to start `docker start netron-mongo`.

## How to run
This will find a network to model `sin(x)`.  
  
1. Start a server: `python server.py --input_dim 1 --output_dim 1 --data sin_data.npz --solver GridSearch --grid simple_params_grid.json`  
2. Start a worker: `python worker.py  --server http://localhost:8080 --nb_epoch 10 --patience 5`  

Check `python server.py -h` and `python worker.py -h` for the explanation of the arguments.  
You can check the results of the training by opening `http://localhost:8080/stats/` in your browser.

## Creating training data for workers
Training data must be stored as a compressed numpy file in
`netron/server/static/`. To create such file for your training data:  
```python
np.savez_compressed("data_filename.npz", X_train = your_x_train, y_train = your_y_train)
```

Where `your_x_train` and `your_y_train` are numpy arrays.

## Contribution
1. Open a separate branch and make your changes there.  
2. Open a pull request to master (For now. It'll change later).


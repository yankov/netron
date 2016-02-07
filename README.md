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

## Contribution
1. Open a separate branch and make your changes there.  
2. Open a pull request to master (For now. It'll change later).


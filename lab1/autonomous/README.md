# Autonomous lab

In this lab we will explore the optimization effects of different algorithms on test functions and simple datasets, and compare to how they work on deep neural networks using the Keras: "Reuters newswire topics classification" dataset.

### References
* [Ian J. Goodfellow and Oriol Vinyals: Qualitatively characterizing neural network optimization problems](https://dblp.org/rec/bib/journals/corr/GoodfellowV14) has some good visualization and comparasions of optimization functions 
* [Daniel Jiwoong Im and Michael Tao and Kristin Branson: An Empirical Analysis of Deep Network Loss Surfaces](http://arxiv.org/abs/1612.04010) also
* [This git repo by Mike Clark](https://github.com/wassname/viz_torch_optim) with animated visualizations which is pretty cool
* [Alec Radford's animations for optimization algorithms](http://www.denizyuret.com/2015/03/alec-radfords-animations-for.html)
* And others that can be found in the reports references.bib

### Running the code
The code to run is lab1.py and has various configuration options for the training of the FNN, these include:
* -w --words: max words to use (default: 1000)
* -b --batch-size: size of batches to use (default: 32)
* -e --epochs: amount of epochs to use (default: 5)
* -o --output_prefix: prefix for the output files (default: empty) this is to store in a different subfolder of /result the results obtained 
* -f --test_function: visualize optimizations using test functions (beale animation) (default: false)

These options can be viewed as well by running the file as:
```bash
python lab1.py --help
usage: lab1.py [-h] [-w WORDS] [-b BATCH_SIZE] [-e EPOCHS] [-o OUTPUT_PREFIX][-f]

Uses the Reuters dataset to compare the results of different optimization functions
    
optional arguments:
    -h, --help show this help message and exit
    -w WORDS, --words WORDS max words to use (default: 1000)
    -b BATCH_SIZE, --batch_size BATCH_SIZE size of batches to use (default: 32)
    -e EPOCHS, --epochs EPOCHS amount of epochs to use (default: 5)
    -o OUTPUT_PREFIX, --output_prefix OUTPUT_PREFIX prefix for the output files (default: empty)
    -f, --test_function visualize optimizations using test functions (default: false)
```

The code creates a visualization of the optimization functions using the make_moons dataset from sklearn, and then goes on to use the Reuters dataset with the network defined and configurations. The model is made using all the optimizers specified ('sgd','adam','rmsprop','adagrad','adamax','adadelta','nadam') using Keras. 


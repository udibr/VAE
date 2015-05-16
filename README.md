# VAE
Example of a Variational-Autoencoder using Theano blocks

Dependencies
------------
 * [Blocks](https://github.com/bartvm/blocks) follow
the [install instructions](http://blocks.readthedocs.org/en/latest/setup.html).
This will install all the other dependencies for you (Theano, Fuel, etc.).

Notes
-----
 * This is a work in progress
 * dropout does not work for now
 
Example
-------
    > python VAE.py --gamma 0.01 --batch_size 1000
        Training status:
         batch_interrupt_received: False
         epoch_interrupt_received: False
         epoch_started: False
         epochs_done: 1000
         iterations_done: 60000
         received_first_batch: True
         training_started: True
    Log records from the iteration 60000:
         saved_to: vaemnist-L784,100,20-l13-g12-b10
         test_nll_bound: 105.619911194
         test_recons_term: 79.463760376
         time_read_data_this_epoch: 0.119498491287
         time_read_data_total: 118.455649614
         time_train_this_epoch: 0.715604543686
         time_train_total: 731.838293791
         train_nll_bound: 106.093231201
         train_total_gradient_norm: 32.678276062
         training_finish_requested: True
         training_finished: True

class ae_param:

    # error counter for user inputs
    max_err = 3         

    # ratio of the test dataset for final assessment
    te_size = 0.3

    # parameters for autoencoder
    act_en1 = "relu"    # activation function for 1st encoder
    act_en2 = "relu"    # activation function for 2nd encoder
    act_de1 = "relu"    # activation function for 1st decoder
    act_de2 = "relu"    # activation function for 2nd decoder
    
    dim_en1 = 5         # number of 1st encoder outputs (dimensions)
    dim_en2 = 3         # number of 2nd encoder outputs
    dim_de1 = 5         # number of 1st decoder outputs

    loss_ae = "binary_crossentropy"     # loss function for autoencoder
    opt_ae = "adam"     # optimizer for autoencoder

    ae_epochs = 10      # training epochs for autoencoder
    vb_ae = 1           # verbosity mode for autoencoder
                          #(0: none, 1: bar, 2: line/epoch)

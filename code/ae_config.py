
#_______________________________________________________________________________
# CE888 Project |     ae_config.py    | Ogulcan Ozer. | Feb. 2019 | UNFINISHED.
#_______________________________________________________________________________

class ae_param:
    
    #Standart AE parameters ###################
    #                                         #
    #Activation functions for standard AE     #
    af_aenc = "linear"                       ##
    af_adec= "linear"                        ##
    #Dimensions                               #
    #                                         #
    #                                         #
    #                                         #
    #                                         #
    #                                         #
    ###########################################
    #Stacked AE parameters ####################
    #                                         #
    #Activation functions for standard AE     #
    saf_aenc1 = "relu"                       ##
    saf_aenc1 = "relu"                       ##
    saf_adec= "relu"                         ##
    saf_adec1 = "relu"                       ##
    #Dimensions                               #
    #                                         #
    #                                         #
    #                                         #
    #                                         #
    #                                         #
    ###########################################
    #Global parameters ########################
    #                                         #
    ae_loss = "mean_squared_error"           ##
    ae_opt = "adam"                          ##
    ae_epoch = 100                           ##
    ae_batch = 128                           ##
    learning_rate = 0.1                      ##
    verbosity = 1                            ##
    ###########################################

#-------------------------------------------------------------------------------
# End of ae_config.py 
#-------------------------------------------------------------------------------

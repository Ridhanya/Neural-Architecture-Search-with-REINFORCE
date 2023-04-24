############ Search Space Constants ################

target_classes = 2


############ Main NAS Constants ################

steps = 500
max_len = 3


############ Controller Constants ################

controller_lstm_dim = 100
controller_lr = 0.01


############ Model generation Constants ################

gen_epochs = 150
mlp_lr = 0.01
mlp_loss_func = 'categorical_crossentropy'
metrics = ['accuracy']
batch_size = 256
mlp_dropout = 0.5



############ LOGS - REINFORCERUN.PY #################
nas_data_log = 'LOGS/generateddata.pkl'
nas_full_data_log = 'LOGS/fulldata.pkl'
from reinforcesearchspace import *
from reinforcevariables import *
from reinforcecontroller import *
from reinforcetrainpolicy import *
from reinforcegeneratemodel import *
import tensorflow as tf


#main search function
def reinforcesearch(input_shape, x_data, y_data, x_val, y_val):
    #call the search space
    vocabsearch = vocab_dict()
    #save the index of the search space
    vocab_idx = [0]+list(vocabsearch.keys())
    #controller model input shape
    controller_input_shape = (1, max_len - 1)
    #controller class
    controller_classes = len(vocabsearch)+1
    #initialize the controller model
    policymodel = controllermodel(controller_lstm_dim, controller_input_shape, len(vocabsearch)+1)
    #data to return
    data = []
    #list to store valid architecture sequence and validation accuracy
    valid_architectures = []
    valid_raw = []
    #initialize policy model
    policymodel = controllermodel(controller_lstm_dim,controller_input_shape,controller_classes)
    #variable to track the number of valid architectures generated over the steps
    #can be useful when have to compare models from different search strategies
    validnum = 0
    #iterate over the steps
    for s in range(steps):
        sequence=[]
        current_state = tf.keras.utils.pad_sequences([sequence], maxlen= max_len - 1, padding='post').reshape(1,1,max_len - 1,1)
        states, rewards, actions = [], [], []
        done = False
        
        #episode terminates if the valid architecture is produced and not produced
        while not done :
            #print("current_state" , current_state)
            prob = np.array(policymodel(current_state)[0][0])
            #print(np.array(prob)/np.array(prob).sum())
            action = np.random.choice(vocab_idx, size=1, p=(prob/prob.sum()))[0]
            actions.append(action)
            #sequence.append(action)
            states.append(current_state)
            #reward = -2
            # print(action)
            # print("sequence ", sequence)
            # print("lenght sequence ", len(sequence))
            
            #check for action validity to determine rewards
            if (action == (len(vocabsearch)-1)) and (len(sequence)) == 0:
                reward = -1
                rewards.append(reward)
                done = True
            elif action == len(vocabsearch) and len(sequence) == 0 :
                reward = -1
                rewards.append(reward)
                done = True
            elif action == 0 and len(sequence)==0:
                reward = -1
                rewards.append(reward)
                done = True
            elif (len(sequence) == max_len -1) and action != len(vocabsearch) :
                reward = -1
                rewards.append(reward)
                done = True
            else:
                if(len(sequence) >=1):
                    if (action < 197 and sequence[-1]>=197):
                        reward = -1
                        rewards.append(reward)
                        done=True
                    else:
                            #conditions to update "done"
                            if len(sequence) < max_len-1 and action != len(vocabsearch) and not action == 0 :
                                reward = 0
                                rewards.append(reward)
                                sequence.append(action)
                            elif action == 0:
                                reward = -1
                                rewards.append(reward)
                                done = True
                            else:
                                #generate the model and train it to get the rewards
                                sequence.append(action)
                                print("sequence in model ", sequence)
                                if sequence not in valid_raw:
                                    model = create_model(sequence, input_shape, vocabsearch)
                                    compilemodel = compile_model(model)
                                    trainmodel_history = train_model(model,x_data,y_data,gen_epochs, x_val, y_val,callbacks=None)
                                    reward = rewardforgenmodel(trainmodel_history)
                                    print("validation accuracy " , reward)
                                    valid_architectures.append([encode_sequences(sequence,vocabsearch),reward])
                                    valid_raw.append(sequence)
                                validnum+=1
                                done=True
                elif not action == 0:
                    reward = 0
                    rewards.append(reward)
                    sequence.append(action)
            if done:
                #train the policy model
                data.append([current_state,action,rewards[-1]])
                trainpolicy(states,rewards, actions,policymodel)
            else:
                #print("seuqnece in done ", sequence)
                data.append([current_state,action,rewards[-1]])
                next_state = tf.keras.utils.pad_sequences([sequence], maxlen= max_len - 1, padding='post').reshape(1,1,max_len - 1,1)
                #print("next state ", next_state)
                current_state = next_state
            #print(data[-1])

            if(validnum == 50):
                break

    return data, valid_architectures

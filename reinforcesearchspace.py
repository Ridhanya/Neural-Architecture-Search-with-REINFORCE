from reinforcevariables import *

def vocab_dict():

        #define conv1d keys and values : {key : (filter size, kernel size)} 
        #for simplicity activation function is always relu and strides and padding are one. 
        filter_size = [3,5,9,11,13,15,17,19,21,23,25,27,29,31]
        kernel_size = [3,5,9,11,13,15,17,19,21,23,25,27,29,31]

        layer_params = []
        layer_id = []
        
        for i in range(len(filter_size)):
            for j in range(len(kernel_size)):


                #create (id (key),filter_size, kernel_size)
                layer_params.append((filter_size[i],kernel_size[j]))
                layer_id.append(len(filter_size)*i+j+1)


        #create a search space dictionary
        vocab = dict(zip(layer_id,layer_params))

        #add fully connected layers to the search space - vocab
        #sticking to relu activation for simplicity
        nodes = [4,8,10,12,16,19,25,50,150,200,300,500]
        act_funcs = ['relu']

        for i in range(len(nodes)):
            for j in range(len(act_funcs)):


                #create (id (key),filter_size, kernel_size)
                layer_params.append((nodes[i],act_funcs[j]))
                vocab[len(vocab)+1] = layer_params[-1]



        #add dropout to the dictionary - for simplicity dropout size is always 0.2
        vocab[len(vocab)+1] = (('dropout'))

        #add the softmax/sigmoid layer to the dictionary
        if target_classes == 2:
            vocab[len(vocab)+1] = (target_classes - 1, 'sigmoid')
        else:
            vocab[len(vocab)+1] = (target_classes , 'softmax')
        
        return vocab

#function to encode an architecture sequence
def encode_sequences(sequence,vocab):
        #keys = list(vocab.keys())
        values = list(vocab.values())
        encoded_sequences = []
        for value in sequence:
            encoded_sequences.append(values[value-1])
        return encoded_sequences
    

#function to decode the sequences
def decode_sequence(sequence, vocab):
        keys=list(vocab.keys())
        values = list(vocab.values())
        decoded_sequence = []
        for key in sequence:
            decoded_sequence.append(values[keys.index(key)])
        return decoded_sequence
from keras.models import load_model
from src import config
import numpy as np
import json
from tqdm import tqdm


def Generating(corpus ,int_char,model,gen_length = 100):
    generate_string = ""
    print("Generating String .................")
    print(" ")
    for _ in tqdm(range(gen_length)):
        y_pre=model.predict(corpus/len(int_char))
        pre = np.argmax(y_pre)
        generate_string += int_char[pre]
        pre = pre.reshape((1,1,1))
        corpus=np.concatenate([corpus,pre],axis=1)
        corpus=corpus[:,1:,:]
    return generate_string

if __name__ == '__main__':
    ## Getting the paths from the configuration file
    test_path = config.test_file_path
    model_path = config.model_path
    seq_length = config.Seq_length
    json_path = "output/char_int.json"
    
    ### opening the json file for the dictonary of characters 
    f = open(json_path,"r")
    char_int = json.load(f)
    int_char = dict((i,j) for j,i in char_int.items())
    ### loading the test file 
    fp = open(test_path,"r",encoding="utf-8")
    model = load_model(model_path)
    
    ## loading and preprocessing the test file
    test_data = fp.read()
    test_data = test_data.lower()
    test_data = test_data[:seq_length]
    print(test_data)
    test_data = [char_int[c] for c in test_data]
    test_data = np.array(test_data).reshape(1,seq_length,1)
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
    ## Generating the new text from the model
    generated_text = Generating(test_data,int_char,model)
    print("=============================GENERATED TEXT==============================")
    print(generated_text)
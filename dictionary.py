import pickle
import numpy as np
import os


class Database:
    def __init__(self):
        if(os.path.isfile("database.pickle")):
            print('File Existing !')
            with open('database.pickle','rb') as handle:
                self.dictionary = pickle.load(handle)
                
        else:
            print('File does not exist, create a new file !')
            self.dictionary = []

    def __call__(self):
        if(self.num_in_dict() == 0):
            print('No one exist !')
           
        return self.dictionary
	
    def name_list(self):
        name_list = len(self.dictionary) * [None]
        for key, value in enumerate(self.dictionary):
            name_list[key] = value[0]
        return name_list

    def feat_list(self):
        feat_list = len(self.dictionary) * [None]
        for key, value in enumerate(self.dictionary):
            feat_list[key] = value[1]
        return feat_list
 
    def num_in_dict(self):
        return len(self.dictionary)   
 
    def add_one(self, name, feat):
        tmp = [name, feat]
        self.dictionary.append(tmp)
        self.save_dict()

    def del_one(self, key):
        del self.dictionary[key]
        self.save_dict()
    
    def edit(self, key, rename):
        self.dictionary[key][0] = rename
        self.save_dict()
   
    def save_dict(self):
        with open('database.pickle','wb') as handle:
            pickle.dump(self.dictionary, handle, protocol = pickle.HIGHEST_PROTOCOL)
'''
dictionary = Database() 
print('__call__', dictionary())
print('num_in_dict', dictionary.num_in_dict())
print('name_list', dictionary.name_list())
print('feat_list', dictionary.feat_list())
print('add_one', dictionary.add_one('Gina',np.array([1000,2000])))
print('num_in_dict', dictionary.num_in_dict())
print('add_one', dictionary.add_one('Gina',np.array([1000,2000])))
print('num_in_dict', dictionary.num_in_dict())
print('__call__', dictionary())
print('del_one', dictionary.del_one(0))
print('num_in_dict', dictionary.num_in_dict())
print('__call__', dictionary())
print('name_list', dictionary.name_list())
print('feat_list', dictionary.feat_list())
'''

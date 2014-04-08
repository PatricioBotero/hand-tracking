'''
Created on 01.04.2014

@author: tabuchte
'''

class pet:
    number_of_legs = 0

    def set_pet_legs(self, n):
        pet.number_of_legs = n
        
    def __str__ (self):
        return 'I\'m a pet!'
    
    def __int__(self):  # operator __int__ is called if an instance of pet is casted to a string
        return 42 

class A:
    
    def __init__(self):
        self.a = None
        
    def set_a(self, a):
        self.a = a
        
        

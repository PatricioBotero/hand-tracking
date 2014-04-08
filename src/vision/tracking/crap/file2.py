'''
Created on 01.04.2014

@author: tabuchte
'''

from file1 import *


dog = pet()
dog.number_of_legs = 1
print (dog.number_of_legs)

      
a = A()
print (a.a)
a.set_a(3)
print(a.a)

b = A()
print (b.a)
b.a = 4
print (b.a)

a = A()
a.a = 70
print (a.a)


pp = pet()
print (pp)
print (int(pp))


pets = []
for i in range(3):
    p = pet()
    p.weight = i
    pets.append(p)
    
for i in range(3):    
    print (str(pets[i]), pets[i].weight)
    
    
for i in range(3):
    p = pets[i]
    p.weight = i**2
    print(str(p), p.weight)
from __future__ import print_function
from six.moves import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import string 

# f  = open('./notmnist_large/b.pickle')
# df = pickle.load(f)
# fig = plt.figure()
# squaredim = 7

# Display a sample of pickled images
# for i in range(0, squareDim ** 2):
#     fig.add_subplot(squareDim,squareDim,i+1)
#     plt.tick_params(
#             axis='both',
#             left='off', 
#             top='off', 
#             right='off', 
#             bottom='off', 
#             labelleft='off', 
#             labeltop='off', 
#             labelright='off', 
#             labelbottom='off'
#     )
#     plt.imshow(df[i,:,:])
# 
# plt.show()

# Verify data are balanced across classes
# for letter in string.ascii_lowercase[:10]: 
#    f = open('./notmnist_large/' + letter + '.pickle') 
#    df = pickle.load(f)
#    print(letter + ': ', end='')
#    print(df.shape)
# a: (52909, 28, 28)
# b: (52911, 28, 28)
# c: (52912, 28, 28)
# d: (52911, 28, 28)
# e: (52912, 28, 28)
# f: (52912, 28, 28)
# g: (52912, 28, 28)
# h: (52912, 28, 28)
# i: (52912, 28, 28)
# j: (52911, 28, 28)


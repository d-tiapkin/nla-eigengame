import numpy as np
import numpy.random as rd
import numpy.linalg as la
import time

DEBUG = True

class Test:
    """
      alg: our algorithm
      
      name: string for print function
        (for example, "01 - small square random matrices")
      
      range: max value of matrix element
      
      style: filling algorithm for random matrix, it causes amount of
      useful size axises (only "size" variable matters with "square" size)
        random;
      size, size2: size params, which may be used in matrix initialization

      isint: boolean variable which is one if matrix elements is intigers
    """

    def __init__(self, alg, name, amount, range, style, size, size2 = 0, isint = False):
        self.amount = amount
        self.style = style
        self.size = size
        self.range = range
        self.size2 = size2

    def testing(self):
        our_time = 0
        numpy_time = 0

        our_losses = []
        numpy_losses = []

        for i in range(amount):
          if(style == "random"):
            matrix = range*(rd.rand(size,size2))
            if(isint):
              matrix = int(matrix)
          
          elif(style == "from_function"):
            matrix = np.zeros(size, size2)
            if(isint):
              matrix = int(matrix)
          
          #Our testing
          start = time.time()
          res1 = la.alg(matrix)
          finish = time.time()
          our_time += finish-start

          #SVD testing
          start = time.time()
          res2 = la.svd(matrix)
          finish = time.time()
          numpy_time += finish-start
          
          #Loss
          our_loss, numpy_loss = loss_evaluation(res1, res2, matrix) 

          #-----------------------------------------------------------------
          if(DEBUG):
            if(i < 5):
              print(matrix[:10, :10])
          #-----------------------------------------------------------------        
        
          
        self.times = (our_time, numpy_time)

    def out(self):        
        print(f"\
        ------------------------------------------------------------------------\
        [{self.name}]\n\
        Our time: {round(our_time, 4)}s.\
        VS NumPy time: {round(numpy_time, 4)}\n\
        (ratio = {round(numpy_time/our_time, 4)})\n\
        iteration: {amount}\n\n\
        Total Loss: {round(our_loss, 4)}\
        VS NumPy Loss: {round(numpy_loss, 4)}\n\
        Mean loss per iteration: {round(our_loss/amount, 4)}\
        VS NumPy: {round(numpy_loss/amount, 4)}\n\
        ------------------------------------------------------------------------\
        ")

#-------------------------------------------------------------------------------

def tests(alg):
  tests_array = [
  Test(alg, "01 - small square random matrices", 1000, 100, "random", 1000, 500, False)
  ]

  for test in tests_array:
    test.testing()
    test.out()








  
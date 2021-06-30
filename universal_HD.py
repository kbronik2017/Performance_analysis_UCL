
#!/usr/bin/env python3

#   Universal Hausdorff Distance computer

#   Copyright (C) 2019 UCL
#   
#   https://www.ucl.ac.uk/medical-physics-biomedical-engineering/
#   Developed   at  Department of Medical Physics and Biomedical Engineering
#   University, UCL, UK
#   

#   This python code uses the following distance functions from scipy.spatial.distance.cdist and compute distance 
#   between each pair of the two collections of inputs, which will be 
#   then later used to compute 2-3D  Mean/Direct Symmetric/Nonsymmetric Hausdorff Distance


# Y = cdist(XA, XB, 'euclidean')

# Computes the distance between  points using Euclidean distance (2-norm) as the distance metric between the points. The points are arranged as  -dimensional row vectors in the matrix X.

# Y = cdist(XA, XB, 'minkowski', p=2.)

# Computes the distances using the Minkowski distance  (-norm) where .

# Y = cdist(XA, XB, 'cityblock')

# Computes the city block or Manhattan distance between the points.

# Y = cdist(XA, XB, 'seuclidean', V=None)

# Computes the standardized Euclidean distance. The standardized Euclidean distance between two n-vectors u and v is

# V is the variance vector; V[i] is the variance computed over all the i’th components of the points. If not passed, it is automatically computed.

# Y = cdist(XA, XB, 'sqeuclidean')

# Computes the squared Euclidean distance  between the vectors.

# Y = cdist(XA, XB, 'cosine')

# Computes the cosine distance between vectors u and v,

# where  is the 2-norm of its argument *, and  is the dot product of  and .

# Y = cdist(XA, XB, 'correlation')

# Computes the correlation distance between vectors u and v. This is

# where  is the mean of the elements of vector v, and  is the dot product of  and .

# Y = cdist(XA, XB, 'hamming')

# Computes the normalized Hamming distance, or the proportion of those vector elements between two n-vectors u and v which disagree. To save memory, the matrix X can be of type boolean.

# Y = cdist(XA, XB, 'jaccard')

# Computes the Jaccard distance between the points. Given two vectors, u and v, the Jaccard distance is the proportion of those elements u[i] and v[i] that disagree where at least one of them is non-zero.

# Y = cdist(XA, XB, 'chebyshev')

# Computes the Chebyshev distance between the points. The Chebyshev distance between two n-vectors u and v is the maximum norm-1 distance between their respective elements. More precisely, the distance is given by

# Y = cdist(XA, XB, 'canberra')

# Computes the Canberra distance between the points. The Canberra distance between two points u and v is

# Y = cdist(XA, XB, 'braycurtis')

# Computes the Bray-Curtis distance between the points. The Bray-Curtis distance between two points u and v is

# Y = cdist(XA, XB, 'mahalanobis', VI=None)

# Computes the Mahalanobis distance between the points. The Mahalanobis distance between two points u and v is  where  (the VI variable) is the inverse covariance. If VI is not None, VI will be used as the inverse covariance matrix.

# Y = cdist(XA, XB, 'yule')

# Computes the Yule distance between the boolean vectors. (see yule function documentation)

# Y = cdist(XA, XB, 'matching')

# Synonym for ‘hamming’.

# Y = cdist(XA, XB, 'dice')

# Computes the Dice distance between the boolean vectors. (see dice function documentation)

# Y = cdist(XA, XB, 'kulsinski')

# Computes the Kulsinski distance between the boolean vectors. (see kulsinski function documentation)

# Y = cdist(XA, XB, 'rogerstanimoto')

# Computes the Rogers-Tanimoto distance between the boolean vectors. (see rogerstanimoto function documentation)

# Y = cdist(XA, XB, 'russellrao')

# Computes the Russell-Rao distance between the boolean vectors. (see russellrao function documentation)

# Y = cdist(XA, XB, 'sokalmichener')

# Computes the Sokal-Michener distance between the boolean vectors. (see sokalmichener function documentation)

# Y = cdist(XA, XB, 'sokalsneath')

# Computes the Sokal-Sneath distance between the vectors. (see sokalsneath function documentation)

# Y = cdist(XA, XB, 'wminkowski', p=2., w=w)

# Computes the weighted Minkowski distance between the vectors. (see wminkowski function documentation)

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html?highlight=cdist#scipy.spatial.distance.cdist= cdist(XA, XB, f) 
#  


       
import os
import sys
import signal
import subprocess
import time
import platform
import nibabel as nib
import numpy as np
from scipy.spatial import distance
import numba 


if len(sys.argv) < 5:
    print("First argument:  lesion file name/path.")
    print("Second argument: manual file name/path.")
    print("Third argument:  distance function.")
    print("Fourth argument: direct/mean.")
    print("Fifth argument: symmetry.")
    print("ERROR:" "Number of the arguments must be at least five,  quiting program.")
    time.sleep(1)
    os.kill(os.getpid(), signal.SIGTERM)


les= sys.argv[1]
man= sys.argv[2]
dist= sys.argv[3]
fact= sys.argv[4]
sym=sys.argv[5]
try:
    weight=sys.argv[6]
except IndexError:
    weight=""
try:
    p_norm=sys.argv[7]
except IndexError:
    p_norm=""

if weight:
    pass
else:
    weight=1.0

if p_norm:
    pass
else:
    p_norm=2.0



# print ("Identifying the inputs required for the calculation ...")
# print ("First argument/input ...> lesion mask:", les)
# print ("Second argument/input ...> manual mask:", man)
# print ("Third argument/input ...> distance function:", dist)
# print ("Fourth argument/input ...> factor:", fact)
# print ("Fifth argument/input ...> symmetry:", sym)
# print ("Sixth argument/input ...> weight:", weight)
# print ("Seventh argument/input ...> p_norm:", p_norm)


def distx(argument, a, b):
    if str(argument)=='euclidean':
       return distance.cdist(a, b, 'euclidean')
    if str(argument)=='minkowski':  
       return distance.cdist(a, b, 'minkowski',  p=float(p_norm))    
    if str(argument)=='cityblock':  
       return distance.cdist(a, b, 'cityblock')
    if str(argument)=='cosine':  
       return distance.cdist(a, b, 'cosine') 
    if str(argument)=='hamming':  
       return distance.cdist(a, b, 'hamming')      
    if str(argument)=='seuclidean':  
       return distance.cdist(a, b, 'seuclidean')    
    if str(argument)=='sqeuclidean':  
       return distance.cdist(a, b, 'sqeuclidean')
    if str(argument)=='correlation':  
       return distance.cdist(a, b, 'correlation')    
    if str(argument)=='jaccard':  
       return distance.cdist(a, b, 'jaccard')
    if str(argument)=='chebyshev':  
       return distance.cdist(a, b, 'chebyshev') 
    if str(argument)=='canberra':  
       return distance.cdist(a, b, 'canberra')
    if str(argument)=='braycurtis':  
       return distance.cdist(a, b, 'braycurtis')    
    if str(argument)=='mahalanobis':  
       return distance.cdist(a, b, 'mahalanobis')
    if str(argument)=='yule':  
       return distance.cdist(a, b, 'yule') 
    if str(argument)=='matching':  
       return distance.cdist(a, b, 'matching')
    if str(argument)=='dice':  
       return distance.cdist(a, b, 'dice')    
    if str(argument)=='kulsinski':  
       return distance.cdist(a, b, 'kulsinski')
    if str(argument)=='rogerstanimoto':  
       return distance.cdist(a, b, 'rogerstanimoto')                     
    if str(argument)=='russellrao':  
       return distance.cdist(a, b, 'russellrao')    
    if str(argument)=='sokalmichener':  
       return distance.cdist(a, b, 'sokalmichener')
    if str(argument)=='sokalsneath':  
       return distance.cdist(a, b, 'sokalsneath') 
    if str(argument)=='wminkowski':  
       return distance.cdist(a, b, 'wminkowski', p=float(p_norm), w=float(weight))
    else:
        print("ERROR:" "Unknown  distance function, quiting program.")
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)
    
@numba.jit(parallel=True)
def parallel_for_mean(arr, iterations):
  result = 0
  for i in range(iterations):
     result += min(arr[i])
  return result/iterations 



def for_direct(arr, iterations):
    result=[]  
    for i in range(iterations):
      result.append(min(arr[i]))
    return result


@numba.jit(parallel=True)
def Universal_Hausdorff_Distance_computer (y_true, y_pred, distance, factors, symm):
    if str(factors)=='direct':
       if str(symm)=='symmetric':
          return UHD_direct_sym (y_true, y_pred, distance)
       else: 
          return UHD_direct_nonsym (y_true, y_pred, distance)
    if str(factors)=='mean':
        if str(symm)=='symmetric':
           return  UHD_mean_sym (y_true, y_pred, distance)
        else:
           return  UHD_mean_nonsym (y_true, y_pred, distance)   
    else:
        print("ERROR:" "Unknown  factor! (must be direct or mean) quiting program.")
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)


@numba.jit(parallel=True)
def UHD_mean_sym(y_true, y_pred, distance):
   #  if type(y_true) and type(y_pred) is not np.ndarray:

   #      input_sequence_ls = nib.load(str(y_pred))

   #      input_image_ls = input_sequence_ls.get_data()
   #      indicesls = np.stack(np.nonzero(input_image_ls), axis=1)
   #      indicesls = [tuple(idx) for idx in indicesls]
   #      input_sequence_mn = nib.load(str(y_true))
   #      input_image_mn = input_sequence_mn.get_data()
   #      indicesmn = np.stack(np.nonzero(input_image_mn), axis=1)
   #      indicesmn = [tuple(idx) for idx in indicesmn]
   #      indicesls = np.asarray(indicesls, dtype=np.float32)
   #      indicesmn = np.asarray(indicesmn, dtype=np.float32)
   #  else:
   #      indicesmn = y_true
   #      indicesls = y_pred
 

   input_sequence_ls = nib.load(str(y_pred))

   input_image_ls = input_sequence_ls.get_data()
   indicesls = np.stack(np.nonzero(input_image_ls), axis=1)
   indicesls = [tuple(idx) for idx in indicesls]
   input_sequence_mn = nib.load(str(y_true))
   input_image_mn = input_sequence_mn.get_data()
   indicesmn = np.stack(np.nonzero(input_image_mn), axis=1)
   indicesmn = [tuple(idx) for idx in indicesmn]
   # Preserve float dtypes, but convert everything else to np.float64
    # for stability.
   indicesls = np.asarray(indicesls, dtype=np.float64)
   indicesmn = np.asarray(indicesmn, dtype=np.float64)


    #  Y= distance.cdist(indicesmn,indicesls, 'euclidean')
    #  L= distance.cdist(indicesls,indicesmn, 'euclidean')

   Y = distx(distance, indicesmn, indicesls)
   L = distx(distance, indicesls, indicesmn)
   atob = parallel_for_mean(Y, Y.shape[0])
   btoa = parallel_for_mean(L, L.shape[0])
   result_hd = np.maximum(atob, btoa)
   return result_hd

@numba.jit(parallel=True)
def UHD_mean_nonsym (y_true, y_pred, distance):

   #  if type(y_true) and type(y_pred) is not np.ndarray:

   #      input_sequence_ls = nib.load(str(y_pred))

   #      input_image_ls = input_sequence_ls.get_data()
   #      indicesls = np.stack(np.nonzero(input_image_ls), axis=1)
   #      indicesls = [tuple(idx) for idx in indicesls]
   #      input_sequence_mn = nib.load(str(y_true))
   #      input_image_mn = input_sequence_mn.get_data()
   #      indicesmn = np.stack(np.nonzero(input_image_mn), axis=1)
   #      indicesmn = [tuple(idx) for idx in indicesmn]
   #      indicesls = np.asarray(indicesls, dtype=np.float32)
   #      indicesmn = np.asarray(indicesmn, dtype=np.float32)
   #  else:
   #      indicesmn = y_true
   #      indicesls = y_pred
 

   input_sequence_ls = nib.load(str(y_pred))

   input_image_ls = input_sequence_ls.get_data()
   indicesls = np.stack(np.nonzero(input_image_ls), axis=1)
   indicesls = [tuple(idx) for idx in indicesls]
   input_sequence_mn = nib.load(str(y_true))
   input_image_mn = input_sequence_mn.get_data()
   indicesmn = np.stack(np.nonzero(input_image_mn), axis=1)
   indicesmn = [tuple(idx) for idx in indicesmn]
   indicesls = np.asarray(indicesls, dtype=np.float64)
   indicesmn = np.asarray(indicesmn, dtype=np.float64)
           
  #  Y= distance.cdist(indicesmn,indicesls, 'euclidean')
  #  L= distance.cdist(indicesls,indicesmn, 'euclidean')
   
   Y= distx(distance, indicesmn, indicesls)
    # L= distx(distance, indicesls, indicesmn)
   atob=parallel_for_mean(Y, Y.shape[0])
    # btoa=parallel_for_mean(L, L.shape[0])
   result_hd=np.max(atob)
   return result_hd



@numba.jit(parallel=True)
def  UHD_direct_sym (y_true, y_pred, distance):
   #  if type(y_true) and type(y_pred) is not np.ndarray:

   #      input_sequence_ls = nib.load(str(y_pred))

   #      input_image_ls = input_sequence_ls.get_data()
   #      indicesls = np.stack(np.nonzero(input_image_ls), axis=1)
   #      indicesls = [tuple(idx) for idx in indicesls]
   #      input_sequence_mn = nib.load(str(y_true))
   #      input_image_mn = input_sequence_mn.get_data()
   #      indicesmn = np.stack(np.nonzero(input_image_mn), axis=1)
   #      indicesmn = [tuple(idx) for idx in indicesmn]
   #      indicesls = np.asarray(indicesls, dtype=np.float32)
   #      indicesmn = np.asarray(indicesmn, dtype=np.float32)
   #  else:
   #      indicesmn = y_true
   #      indicesls = y_pred
 

   input_sequence_ls = nib.load(str(y_pred))

   input_image_ls = input_sequence_ls.get_data()
   indicesls = np.stack(np.nonzero(input_image_ls), axis=1)
   indicesls = [tuple(idx) for idx in indicesls]
   input_sequence_mn = nib.load(str(y_true))
   input_image_mn = input_sequence_mn.get_data()
   indicesmn = np.stack(np.nonzero(input_image_mn), axis=1)
   indicesmn = [tuple(idx) for idx in indicesmn]
   indicesls = np.asarray(indicesls, dtype=np.float64)
   indicesmn = np.asarray(indicesmn, dtype=np.float64)
  #  Y= distance.cdist(indicesmn,indicesls, 'euclidean')
  #  L= distance.cdist(indicesls,indicesmn, 'euclidean')
   Y= distx(distance, indicesmn, indicesls)
   L= distx(distance, indicesls, indicesmn)
   atob=for_direct(Y, Y.shape[0])
   btoa=for_direct(L, L.shape[0])
   result_hd=np.maximum(max(atob),max(btoa))
   return result_hd

@numba.jit(parallel=True)
def  UHD_direct_nonsym (y_true, y_pred, distance):
   #  if type(y_true) and type(y_pred) is not np.ndarray:

   #      input_sequence_ls = nib.load(str(y_pred))

   #      input_image_ls = input_sequence_ls.get_data()
   #      indicesls = np.stack(np.nonzero(input_image_ls), axis=1)
   #      indicesls = [tuple(idx) for idx in indicesls]
   #      input_sequence_mn = nib.load(str(y_true))
   #      input_image_mn = input_sequence_mn.get_data()
   #      indicesmn = np.stack(np.nonzero(input_image_mn), axis=1)
   #      indicesmn = [tuple(idx) for idx in indicesmn]
   #      indicesls = np.asarray(indicesls, dtype=np.float32)
   #      indicesmn = np.asarray(indicesmn, dtype=np.float32)
   #  else:
   #      indicesmn = y_true
   #      indicesls = y_pred
 

   input_sequence_ls = nib.load(str(y_pred))

   input_image_ls = input_sequence_ls.get_data()
   indicesls = np.stack(np.nonzero(input_image_ls), axis=1)
   indicesls = [tuple(idx) for idx in indicesls]
   input_sequence_mn = nib.load(str(y_true))
   input_image_mn = input_sequence_mn.get_data()
   indicesmn = np.stack(np.nonzero(input_image_mn), axis=1)
   indicesmn = [tuple(idx) for idx in indicesmn]
   indicesls = np.asarray(indicesls, dtype=np.float64)
   indicesmn = np.asarray(indicesmn, dtype=np.float64)
  #  Y= distance.cdist(indicesmn,indicesls, 'euclidean')
  #  L= distance.cdist(indicesls,indicesmn, 'euclidean')
   Y= distx(distance, indicesmn, indicesls)
    # L= distx(distance, indicesls, indicesmn)
   atob=for_direct(Y, Y.shape[0])
    # btoa=for_direct(L, L.shape[0])
   result_hd=np.max(atob)
   return result_hd



if __name__ == '__main__':

# Universal_Hausdorff_Distance_computer(man, les, dist, fact, sym)

   try:
       print (Universal_Hausdorff_Distance_computer(man, les, dist, fact, sym))
       time.sleep(2)
   except KeyboardInterrupt:
       print("KeyboardInterrupt has been caught.")
       time.sleep(1)
       os.kill(os.getpid(), signal.SIGTERM)

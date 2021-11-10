import numpy as np

def localmap_downsample(localmap, ifonehot = True, zipsize = 10):
    mapsize = len(localmap)
    ratio = int(mapsize/zipsize)
    if(ratio<=1):
        print("error in downsampling localmap")
        return 
    localMap_zip = []
    for ii in range(zipsize):
        for jj in range(zipsize):
            block = localmap[ratio*ii: ratio*(ii+1), ratio*jj: ratio*(jj+1)]
            if (1 in block): # obstacle
                localMap_zip.append(1)
            elif (2 in block): # unobserved
                localMap_zip.append(2)
            else: # free
                localMap_zip.append(0)
    localMap_zip = np.array(localMap_zip)  # (localMap_zip 100*1)
    localMap_zip_2D = localMap_zip.reshape(zipsize, zipsize)  # (localMap_zip_2D 10*10)
    if ifonehot:
        l = len(localMap_zip)
        localMap_onehot = np.zeros(2*l)
        for k in range(len(localMap_zip)):
            if localMap_zip[k] == 0:  # free
                localMap_onehot[k] = 1
            if localMap_zip[k] == 1:  # occupied
                localMap_onehot[k+l] = 1
            # otherwise: unobserved
        return localMap_onehot, localMap_zip_2D
    return localMap_zip, localMap_zip_2D

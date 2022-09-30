import numpy as np
import pylab as pl
from numba import jit

def diffusive_inpaint_freq(tmap,mask,n_inpaint):

    ret = np.zeros(tmap.shape)

    for i in range(0,tmap.shape[2]):

        ret[:,:,i] = diffusive_inpaint(tmap[:,:,i],mask,n_inpaint)

    #    pl.imshow(ret[:,:,i])
    #    pl.show()

    return ret

@jit
def diffusive_inpaint(image,mask,n_inpaint):

    nx = len(image)

    if np.array_equal(np.ones((nx,nx)),mask) == True:

        inpainted_image = image

    else:

        masked_image = image*mask
        zeros = np.where(mask == 0.)
        x_0 = zeros[0]
        y_0 = zeros[1]
        i = 0
        inpainted_image = masked_image

        for j in range(0,n_inpaint,1):

            for i in range(0,len(x_0),1):

                x = x_0[i]
                y = y_0[i]
                c = 0
                value = 0.

                if nx > x-1 >= 0 and nx > y-1 >= 0:

                    value += masked_image[x-1,y-1]
                    c += 1

                if nx > y-1 >= 0:

                    value += masked_image[x,y-1]
                    c += 1

                if nx > x+1 >= 0 and nx > y-1 >= 0:

                    value += masked_image[x+1,y-1]
                    c += 1

                if nx > x-1 >= 0:

                    value += masked_image[x-1,y]
                    c += 1

                if nx > x+1 >= 0:

                    value += masked_image[x+1,y]
                    c += 1

                if nx > x-1 >= 0 and nx > y+1 >= 0:

                    value += masked_image[x-1,y+1]
                    c += 1

                if nx > y+1 >= 0:

                    value += masked_image[x,y+1]
                    c += 1

                if nx > x+1 >= 0 and nx > y+1 >= 0:

                    value += masked_image[x+1,y+1]
                    c += 1

                inpainted_image[x,y] = value/c

    return inpainted_image

import numpy as np
#x = np.random.randint(0,10,(10, 3))
x = np.array( [
        [1,2,3],
        [4,5,6],
        [7,8,9],
        [10,11,12] ] )
y = np.array([1,2,3,4])

print('x=',x)
print('y=',y)
#x_split = np.split(x, 2, axis=0)

indices= np.where(y%2==0)
print('indices=',indices[0])

rp=x.take(indices[0])

print()

exit(0)


print(x_split)
k = 0
print(np.concatenate(x_split[:k] + x_split[k+1:], axis=0))

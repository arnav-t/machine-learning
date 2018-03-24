import numpy as np 
import matplotlib.pyplot as plt 

iters = 1500
alpha = 0.1
file = 'ex1data2.txt'
errors = np.zeros((0,0))

if __name__ == '__main__':
	import sys
	file = sys.argv[1]
	iters = int(input("Enter the number of iterations: "))
	alpha = float(input("Enter the learning rate: "))


def loadData(path):
	data = np.loadtxt(path, delimiter = ',')
	return data

def h():
	return np.matmul(X,theta)

def J():
	term = h() - Y
	Jmat = np.matmul(np.transpose(term),term)/(2*rows)
	return Jmat

def gradientDescent():
	newTheta = theta - (alpha/rows)*np.matmul(np.transpose(X),h()-Y)
	return newTheta

def normalize(A):
	Amean = np.mean(A)
	Amax = np.max(A)
	Amin = np.min(A)
	An = (A - Amean)/(Amax - Amin)
	return An


data = loadData(file)
rows = data.shape[0]
cols = data.shape[1]

Y = np.zeros((rows,1))
Y[:,0] = data[:,cols-1]
Y = normalize(Y)

X = np.ones(data.shape)
i = 0
while(i < cols-1):
	X[:,i+1] = data[:,i]
	X[:,i+1] = normalize(X[:,i+1])
	i += 1

theta = np.zeros((cols,1))

print('Performing gradient descent...')

i = 1
while i <= iters:
	print('Iteration: ' + str(i), end = ' ')
	errors = np.insert(errors,i-1,J()[0,0])
	print('Error: ' + str(errors[i-1]))
	theta = gradientDescent()
	i += 1
	
print('Plotting...')

plt.subplot(cols,1,1)
x = np.zeros((1,iters))
x[0,:] = np.arange(1,iters+1)
y = np.zeros((1,iters))
y[0,:] = errors
plt.plot(x,y,'r,-')
plt.title('Error function')
plt.ylabel('Error')
plt.xlabel('Iterations')

i = 0
while i < cols - 1:
	plt.subplot(cols,1,i+2)
	plt.plot(X[:,i+1],Y,'rx')
	y= np.zeros((rows,1))
	y[:,0] = theta[0,0] + theta[i+1,0]*X[:,i+1]
	x = np.zeros((rows,1))
	x[:,0] = X[:,i+1]
	plt.plot(x,y)
	plt.title('Data Set ' + str(i+1))
	plt.ylabel('Y')
	plt.xlabel('X' + str(i+1))
	i += 1

plt.show()

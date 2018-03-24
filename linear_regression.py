import numpy as np 
import matplotlib.pyplot as plt 

iters = 1500
alpha = 0.01
errors = np.zeros((0,0))

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

data = loadData('ex1data2.txt')
rows = data.shape[0]
cols = data.shape[1]

Y = np.zeros((rows,1))
Y[:,0] = data[:,cols-1]

X = np.ones(data.shape)
i = 0
while(i < cols-1):
	X[:,i+1] = data[:,i]
	i += 1

theta = np.zeros((cols,1))

print('Performing gradient descent...')

i = 1
while i <= iters:
	print('Iteration: ' + str(i))
	errors = np.insert(errors,i-1,np.sum(J()))
	theta = gradientDescent()
	i += 1
	
plt.subplot(cols,1,1)
x = np.zeros((1,iters))
x[0,:] = np.arange(1,iters+1)
y = np.zeros((1,iters))
y[0,:] = errors
plt.plot(x,y)
plt.title('Error function')
plt.ylabel('Error')
plt.xlabel('Iterations')

print('Plotting...')

i = 0
while i < cols - 1:
	plt.subplot(cols,i+2,1)
	plt.plot(X[:,i],Y,'rx')
	x = np.zeros((1,2))
	x[0,0] = np.min(X[:,i])
	x[0,1] = np.max(X[:,i])
	y= np.zeros((1,2))
	y[0,0] = theta[0,0] + theta[i,0]*x[0,0]
	y[0,1] = theta[0,0] + theta[i,0]*x[0,1]
	plt.plot(x,y)
	plt.title('Data Set ' + str(i+1))
	plt.ylabel('Y' + str(i+1))
	plt.xlabel('X' + str(i+1))
	i += 1

plt.show()

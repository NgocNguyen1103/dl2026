data = []

with open('lr.csv', 'r') as file:
    for line in file:
        area, price = line.split(',')
        data.append([float(area), float(price)])

x_list = []
y_list = []
for _ in range(len(data)):
    x_list.append(data[_][0])
    y_list.append(data[_][1])
    print(data[_])

def grad(learning_rate, w0, w1, total_loss):
    while True:
        total_loss = 0
        for _ in range(len(data)):
            x = data[_][0]
            y = data[_][1]
            w0 = w0 - learning_rate*((w1 * x) + w0 - y)
            w1 = w1 - learning_rate*(x * (w1 * x + w0 - y))
            point_loss = ((w1 * x + w0 - y)**2)/2
            total_loss = total_loss + point_loss
        total_loss = total_loss/len(data)
        print(f'loss: {total_loss}')
        print(f'w0: {w0:.3f}, w1: {w1:.3f}')
        if total_loss < 10:
            return w0, w1
w0, w1 = grad(0.0001, 0, 1, 0)


predict = []
for _ in range(len(data)):
    x = data[_][0]
    y = x*w1 + w0
    print(x,y)
    predict.append(y)

import matplotlib.pyplot as plt

plt.scatter(x_list, y_list)
plt.plot(x_list, predict, color='r')
plt.show()
data = []

with open('loan2.csv', 'r') as file:
    for line in file:
        experience, salary, loan = line.split(',')
        data.append([float(experience), float(salary), float(loan)])


for _ in range(len(data)):
    print(data[_])


def ln(x):
    n = 1000.0
    return n * ((x ** (1/n)) - 1)

def exp(x):
    e = 2.718281828459045
    return e**x

def sigmoid(x):
    if x < 0.5:
        return 0
    else: return 1

def w0_df(x1, x2, y, w0, w1, w2):
    inner_sigmoid = 0 - (w1*x1 + w2*x2 + w0)
    inner_sigmoid = sigmoid(inner_sigmoid)
    loss = 1 - y - inner_sigmoid
    return loss

def w1_df(x1, x2, y, w0, w1, w2):
    inner_sigmoid = 0 - (w1*x1 + w2*x2 + w0)
    inner_sigmoid = sigmoid(inner_sigmoid)
    loss = (0-y)*x1 + x1*(1 - inner_sigmoid)
    return loss

def w2_df(x1, x2, y, w0, w1, w2):
    inner_sigmoid = 0 - (w1*x1 + w2*x2 + w0)
    inner_sigmoid = sigmoid(inner_sigmoid)
    loss = (0-y)*x2 + x2*(1 - inner_sigmoid)
    return loss

def single_loss_value(x1, x2, y, w0, w1, w2):
    loss = (0 - y)*(w1 * x1 + w2 * x2 + w0) + ln(1 + exp(w1*x1 + w2*x2 +w0))
    return loss

def grad(learning_rate, w0, w1, w2, total_loss):
    while True:
        loss = 0
        for _ in range(len(data)):
            x1 = data[_][0]
            x2 = data[_][1]
            y = data[_][2]
            w0 = w0 - learning_rate*w0_df(x1, x2, y, w0, w1, w2)
            w1 = w1 - learning_rate*w1_df(x1, x2, y, w0, w1, w2)
            w2 = w2 - learning_rate*w2_df(x1, x2, y, w0, w1, w2)
            single_lost = single_loss_value(x1, x2, y, w0, w1, w2)
            loss = loss + single_lost
        total_loss = loss/len(data)
        print(f'w0: {w0:.3f} | w1: {w1:.3f} | w2: {w2:.3f}')
        print(f'loss: {total_loss}')
        if total_loss < 0.6:
            return w0, w1, w2
            

w0, w1, w2 = grad(0.0001, 0, 1, 2, 0)
for _ in range(len(data)):
    x1 = data[_][0]
    x2 = data[_][1]
    y = sigmoid(w1*x1 + w2*x2 + w0)
    print(x1, x2, y)


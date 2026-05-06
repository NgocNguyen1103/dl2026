def f_x(x):
    return pow(x,2)

def df_x(x):
    return x*2

x = 10
learning_rate = 0.01
i = 0
while True:
    print(f'epoch: {i+1} | x: {x:.3f} | f(x): {f_x(x):.3f}')
    x = x - learning_rate*df_x(x)
    i = i + 1
    if f_x(x) <= 0.001:
        break



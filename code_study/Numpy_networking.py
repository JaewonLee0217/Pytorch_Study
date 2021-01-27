import numpy as np

batch_size, D_in, H, D_out = 64, 1000, 100, 10

x = np.random.randn(batch_size, D_in)
y = np.random.randn(batch_size,D_out)

#무작위로 가중치 초기화
w1 = np.random.randn(D_in,H)
w2 = np.random.randn(H,D_out)

learing_rate = 1e-6
for t in range(500):
    #순전파 단계 : 예측값 y를 계산 즉 y_pred을 계산
    h = x.dot(w1) # 내적 64,100
    h_relu = np.maximum(h,0) # 음수 제거
    y_pred = h_relu.dot(w2) # 64,10

    #손실(loss)을 계산하고 출력한다
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    #손실에 따른 w1, w2의 변화도를 계산하고 역전파한다.
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred) #100,64 * 64,10 => 100,10
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h<0] = 0
    grad_w1 = x.T.dot(grad_h)

    #가중치를 갱신한다
    w1 -=learing_rate * grad_w1
    w2 -= learing_rate * grad_w2


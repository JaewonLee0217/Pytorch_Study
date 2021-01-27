import torch

dtype = torch.float
device = torch.device("cpu")

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device, dtype = dtype)
y = torch.randn(N, D_out, device= device, dtype = dtype)

#이렇게 무작위 x,y정하고 가중치도 초기화

w1 = torch.randn(D_in, H, device = device, dtype = dtype)
w2 = torch.randn(H, D_out, device = device, dtype = dtype)

learing_rate = 1e-6
for t in range(500):
    #순전파 단계 : 예측값 y를 계산한다
    h = x.mm(w1) # 행렬의 곱셈은 .mm으로 연산해 준다
    h_relu = h.clamp(min=0) # clamp로 ReLU함수(활성함수)를 구현.
    y_pred = h_relu.mm(w2) # 다시 w2랑 연산

    loss = (y_pred - y).pow(2).sum().item()
    if t%100 ==99:
        print(t, loss)

    #손실에 따른 w1, w2의 변화도를 계산하고 역전파하는거

    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h<0] = 0
    grad_w1 = x.t().mm(grad_h)

    #경사하강법을 사용하여 가중치를 갱신
    w1 -= learing_rate * grad_w1
    w2 -= learing_rate * grad_w2

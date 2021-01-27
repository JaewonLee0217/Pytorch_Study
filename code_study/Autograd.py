#Pytorch tensor와 autograd
#대규모 복잡한 신경망에서 역전파 단계를 구현하는 것은 아주 어려운 일이 될 수 있으므로

#자동미분 (Autograd)을 사용하여 신경망에서 역전파 단계의 연산을 자동화 할 수 있다

#Autograd를 사용할 때, 신경망의 순전파 단계는 연산 그래프를 정의하게 되고
# 이 그래프의 노드는 텐서, 엣지는 입력 tensor로 부터 출력 tensolr를 만들어내는 함수가 된디ㅏㅣ
# graph를 통해 역전파를 하게 되면 변화도를 쉽게 계산할 수 이싿.
import torch
dtype = torch.float
device = torch.device("cpu")

N, D_in, H, D_out = 64, 1000, 100, 10

# requires_grad를 False로 설정하여 역전파 중에 이 Tensor들에 대한 변화도를 계산할 필요가 없음을 나타낸다.
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

w1 = torch.randn(D_in, H, device=device, dtype = dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device= device, dtype=dtype, requires_grad = True)
#그러니까, 가중치에 대해서 는 역전파중에 이 Tennsor들에 대한 변화도를 계산할 필요가 있다라고 requires_grad = True

learning_rate = 1e-6
for t in range(500):
    # 순전파단계 : Tensor 연산을 사용하여 예상되는 y값을 계산하다. 이는 Tensor를 사용한 순전파 단꼐와 완전히 동일하지만,
    # 역전파 단계를 별도로 구현하지 않아도 되므로
    # 중간값들에 대한 참조를 갖고 있을 필요가 없다.
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    if t %100 == 99:
        print(t, loss.item())
    loss.backward()
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        #가중치 갱신 후에는 수동으로 변화도를 0으로 만듭니다
        w1.grad.zero_()
        w2.grad.zero_()

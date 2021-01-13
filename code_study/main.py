import torch
import numpy as np
#empty
x = torch.empty(5,5)
print(x)

#ones
y = torch.ones(5,5)
print(y)
#zeros
z = torch.zeros(3,3)
print(z)

#rand
a = torch.rand(4,2)
print(a)

#torch.tensor
b = torch.tensor([[1.1,2.2],[2.2,3.3]])
print(b)

#x.size
q = torch.tensor([[1.1,2.2,4.4],[2.2,3.3,5.5]])
print(q.size())

#type(x)
w = torch.tensor([[1.1,2.2,4.4],[2.2,3.3,5.5]])
print(type(w))

x = torch.rand(2,2)
y = torch.rand(2,2)
print(y[1,1])
print(y[:,1])

#torch.add(x,y)
#y.add(x)
#y.add_(x)
y[1,1]
y[:,1]
x= torch.rand(8,8)
print(x.view(64))
# x.view(4,16)
#x.view(-1,16) -> -1은 16나누는걸로 고정된 4값이 들어감
#x.view(-1,4,4) -> CNN 컨볼류션 연산을 하다가 마지막 fully connected layer에서 펴줄때
x = torch.rand(2,2)
print(type(x))
y = x.numpy()
print(type(y))
#item -> tensor가 스칼라 값일 떄 사용가능
x = torch.ones(1)
print(x)
print(x.item())
#x.item() -> loss함수 계산 값을 뽑아 내고 싶을 때



#x = torch.zeros(5, 3, dtype=torch.long)


# 데이터로부터 Tensor를 직접 생성한다

#라는 변수가 대체된다. _ implace

'''
랜덤한 사인곡선 그리기
'''

import math
import torch
import matplotlib.pyplot as plt

# -pi부터 pi 사이에서 점 1000개 추출
x = torch.linspace(-math.pi, math.pi, 1000)

#실제 사인곡선에서 추출한 값으로 y 만들기
y = torch.sin(x)

a = torch.randn(())
b = torch.randn(())
c = torch.randn(())
d = torch.randn(())

#사인함수를 근사할 3차 다항식 정의
y_random = a*x**3 + b*x**2 + c*x + d

#학습률 정의
learnin_rate = 1e-6

for epoch in range(2000):
    y_pred = a*x**3 + b*x**2 + c*x + d

    loss = (y_pred-y).pow(2).sum().item()   #손실 정의
    if epoch % 100 == 0:
        print(f"epoch{epoch+1} loss:{loss}")
    
    grad_y_pred = 2.0*(y_pred -y)   #기울기의 미분값
    grad_a = (grad_y_pred * x ** 3).sum()
    grad_b = (grad_y_pred * x ** 2).sum()
    grad_c = (grad_y_pred * x).sum()
    grad_d = grad_y_pred.sum()

    #가중치 업데이트
    a-=learnin_rate*grad_a
    b-=learnin_rate*grad_b
    c-=learnin_rate*grad_c
    d-=learnin_rate*grad_d


#실제 사인곡선을 실제 y값으로
plt.subplot(3,1,1)
plt.title("y true")
plt.plot(x, y)

#예측한 가중치의 사인 곡선 그리기
plt.subplot(3,1,2)
plt.title("y pred")
plt.plot(x, y_pred)

#랜덤한 가중치의 사인 곡선 그리기
plt.subplot(3,1,3)
plt.title("y random")
plt.plot(x, y_random)

plt.show()
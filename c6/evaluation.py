import matplotlib.pyplot as plt

from rnn import *

loader = DataLoader(dataset, batch_size = 1)

preds = []

total_loss = 0

with torch.no_grad():
    model.load_state_dict(torch.load("rnn.pth", map_location=device))

    for data, label in loader:
        h0 = torch.zeros(5, data.shape[0], 8).to(device)

        #모델 예측값
        pred = model(data.type(torch.FloatTensor).to(device), h0)
        preds.append(pred.item())

        loss = nn.MSELoss()(pred, label/type(torch.FloatTensor).to(device))

        total_loss += loss/len(loader)

total_loss.item()

plt.plot(preds, label="prediction")
plt.plot(dataset.label[30:], label="actual")
plt.legend()
plt.show()
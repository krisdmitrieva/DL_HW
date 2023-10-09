import torch

# будем предсказывать уровень вреда продукта питания для зубов
# в качестве параметров возьмем сладость, кислотность и твердость продукта
# в датасете будут следующие объекты: леденец, апельсиновый фреш, соевый соус, вареное яйцо, огурец

candy = torch.tensor([[0.9, 0.4, 0.9]])
fresh_orange_juice = torch.tensor([[0.7, 0.8, 0.2]])
soy_sauce = torch.tensor([[0.3, 0.9, 0.1]])
egg = torch.tensor([[0.1, 0.1, 0.4]])
cucumber = torch.tensor([[0.3, 0.1, 0.7]])

dataset = [
    (candy, torch.tensor([[0.9]])),
    (fresh_orange_juice, torch.tensor([[0.8]])),
    (soy_sauce, torch.tensor([[0.7]])),
    (egg, torch.tensor([[0.1]])),
    (cucumber, torch.tensor([[0.2]])),
]

torch.manual_seed(81023)

weights = torch.rand((1, 3), requires_grad=True)
bias = torch.rand((1, 1), requires_grad=True)

mse_loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.SGD([weights, bias], lr=1e-2)


def predict_harm_score(obj: torch.Tensor) -> torch.Tensor:
    return obj @ weights.T + bias


def calc_loss(predicted_value: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    return mse_loss_fn(predicted_value, ground_truth)


num_epochs = 10

for i in range(num_epochs):
    for x, y in dataset:
        optimizer.zero_grad()
        harm_score = predict_harm_score(x)

        loss = calc_loss(harm_score, y)
        loss.backward()
        print(loss)
        optimizer.step()

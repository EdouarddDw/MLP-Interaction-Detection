import torch
from multilayer_perceptron import MLP
from synth import get_data, functions
from config import EXPERIMENTS
device = if torch.cuda.is_available() : 'cuda' else 'mps'

# architecure of the model
parameters = [
    'num_features' = 10,
    "hidden_units" = [64, 64],
    "use_main_effect_nets"= False,
    "main_effect_net_units"=[10, 10, 10]
]

'''need to add:
    - dropout logic
    - make optimizer and l2 work with congif.py
    - add data data_loader
    - add snapshot logic and make it save properly in neet file for each function and experients.
    - the double loop to train for every function and every experiment
''''

def train(
    net,
    data_loaders,
    criterion=nn.MSELoss(reduction="mean"),
    nepochs=100,
    verbose=True,
    dropout,
    L2 = False,
    l1_const=1e-4,
    l2_const=0,
    learning_rate=0.01,
    opt_func,
    snapshots = True,
    snap_every = 5,
    device=torch.device(device),
):
    optimizer = opt_func(net.parameters(), lr=learning_rate, weight_decay=l2_const)

    def evaluate(net, data_loader, criterion, device):
        losses = []
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            loss = criterion(net(inputs), labels).cpu().data
            losses.append(loss)
        return torch.stack(losses).mean()

    best_loss = float("inf")
    best_net = None



    for epoch in range(nepochs):
        running_loss = 0.0
        run_count = 0
        for i, data in enumerate(data_loaders["train"], 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels).mean()

            reg_loss = 0
            for name, param in net.named_parameters():
                if "interaction_mlp" in name and "weight" in name:
                    reg_loss += torch.sum(torch.abs(param))
            (loss + reg_loss * l1_const).backward()
            optimizer.step()
            running_loss += loss.item()
            run_count += 1

        if epoch % 1 == 0:
            key = "val" if "val" in data_loaders else "train"
            val_loss = evaluate(net, data_loaders[key], criterion, device)

            if epoch % 2 == 0:
                if verbose:
                    print(
                        "[epoch %d, total %d] train loss: %.4f, val loss: %.4f"
                        % (epoch + 1, nepochs, running_loss / run_count, val_loss)
                    )

            prev_loss = running_loss
            running_loss = 0.0

    if "test" in data_loaders:
        key = "test"
    elif "val" in data_loaders:
        key = "val"
    else:
        key = "train"
    test_loss = evaluate(net, data_loaders[key], criterion, device).item()

    if verbose:
        print("Finished Training. Test loss: ", test_loss)

    return net, test_loss




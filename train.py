import torch
from multilayer_perceptron import MLP
from synth import get_data, functions
from config import EXPERIMENTS
device = if torch.cuda.is_available() : 'cuda' else 'mps'


    # basic parameters: --------------------------
l2_const_set = 0.10
dropout_rate = 0.2
learning_rate_set = 0.01
n_epochs_set = 100
#--------------------------------------------


# architecure of the model
parameters = [
    'num_features' = 10,
    "hidden_units" = [64, 64],
    "use_main_effect_nets"= False,
    "main_effect_net_units"=[10, 10, 10]
]

'''need to add:
    - dropout logic [X]
    - make optimizer and l2 work with congif.py [X]
    - add data data_loader []
    - add snapshot logic and make it save properly in neet file for each function and experients.
    - just the double loop to train for every function and every experiment
''''

def train(
    net,
    data_loaders,
    criterion=nn.MSELoss(reduction="mean"),
    nepochs=n_epochs_set,
    verbose=True,
    L2 = False,
    l1_const=1e-4,
    learning_rate=learning_rate_set,
    opt_func,
    snapshots = True,
    snap_every = 5,
    device=torch.device(device),
):
    if L2:
        l2_const = l2_const_set
    else:
        l2_const = 0

    if opt_func = 'sdg':
        optimizer = nn.optimizer.sdg(net.parameters(), lr=learning_rate, weight_decay=l2_const)
    if opt_func = 'adam':
        optimizer = nn.optimizer.adam(net.parameters(), lr=learning_rate, weight_decay=l2_const)


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




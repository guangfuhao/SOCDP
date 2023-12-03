from configs import *
from models import *
import os
import torch
from torch import nn, optim
import pickle


def train(datasets_train, datasets_test, train_idx=1, model_name='',
          class_num=10, lambda_sparse=0.01, out_dir='', model_config=[]):
    print('Training Index:', train_idx)
    middle_size = model_config[2]
    mnist_net = initialize_model(model_name, class_num, model_config, lambda_sparse)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # print(os.path)
    mnist_net = mnist_net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mnist_net.parameters(), lr=LR, momentum=MOMENTUM)

    train_loss, train_accs, test_accs = train_and_evaluate_model(mnist_net,
                                                                 datasets_train,
                                                                 datasets_test,
                                                                 optimizer,
                                                                 loss_fn,
                                                                 device,
                                                                 model_name, middle_size, class_num, train_idx, out_dir)
    save_model(mnist_net, middle_size, class_num, model_name, train_idx, out_dir=out_dir)
    save_metrics(train_loss, middle_size, train_accs, test_accs, class_num, model_name, train_idx, out_dir=out_dir)

    return train_loss, train_accs, test_accs


def initialize_model(model_name, class_num, model_config, lambda_sparse):
    if model_name == MODEL_NAME[3] or model_name == MODEL_NAME[4]:
        return globals()[model_name](class_num, model_config, lambda_sparse=lambda_sparse)
    else:
        return globals()[model_name](class_num, model_config)


def train_and_evaluate_model(model, train_loader, test_loader, optimizer, loss_fn, device, model_name,
                             middle_size, class_num, train_idx, out_dir):
    train_accs = []
    train_loss = []
    test_accs = []

    for epoch in range(EPOCH):
        running_loss, train_correct = train_one_epoch(model, train_loader, optimizer, loss_fn, device, model_name)
        test_correct = evaluate_model(model, test_loader, device, model_name)
        running_loss /= 60000  # Assuming dataset size is 60000
        train_loss.append(running_loss)
        train_accs.append(train_correct / 60000)
        test_accs.append(test_correct)
        print(f"Epoch: {epoch + 1}, Loss: {running_loss:.4f}, "
              f"Train Accuracy: {train_correct / 60000:.4f}, Test Accuracy: {test_correct:.4f}")
        save_model(model, middle_size, class_num, model_name, train_idx, epoch, out_dir)

    return train_loss, train_accs, test_accs


def train_one_epoch(model, loader, optimizer, loss_fn, device, model_name):
    model.train()
    running_loss = 0.0
    train_correct = 0

    for batch_X, batch_Y in loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = loss_fn(outputs, batch_Y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == batch_Y).sum().item()

    return running_loss, train_correct


def evaluate_model(model, loader, device, model_name):
    model.eval()
    test_correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()

    return test_correct / len(loader.dataset)


def save_model(model, middle_size, class_num, model_name, train_idx, epoch, out_dir='200'):
    filename = f"model_{train_idx}_epoch_{epoch}.pt"
    path = get_path(out_dir+"model_weights", middle_size, class_num, model_name, filename)
    # print(path)
    ensure_dir_exists(path)
    torch.save(model.state_dict(), path)


def save_metrics(train_loss, middle_size, train_accs, test_accs, class_num, model_name, train_idx, out_dir='200'):
    path = get_path_metrics(out_dir+"loss_acc", middle_size, class_num, model_name, EPOCH, train_idx, extension=f".pickle")
    ensure_dir_exists(path)
    save_list_to_pickle([train_loss, train_accs, test_accs], path)


def get_path(base_dir, middle_size, class_num, model_name, identifier, extension=""):
    parts = [
        base_dir,
        f"middle_size_{middle_size}",
        f"class_num_{class_num}",
        model_name,
        f"{model_name}_model_{identifier}{extension}"
    ]
    return os.path.join(*parts)


def get_path_metrics(base_dir, middle_size, class_num, model_name, identifier, train_idx, extension=""):
    parts = [
        base_dir,
        f"middle_size_{middle_size}",
        f"class_num_{class_num}",
        model_name,
        f"{model_name}_model_{identifier}ep_loss_train_test_{train_idx}{extension}"
    ]
    return os.path.join(*parts)


def ensure_dir_exists(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_list_to_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

# -*- coding: utf-8 -*-
import time

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from alexnet import AlexNet
from utils import train_dir, validation_dir, TestImgData, test_dir, get_labels


def validate_model(model, test_data, criterion=torch.nn.NLLLoss()):
    """
    Test model classification accuracy.
    :param model: the model for testing
    :param test_data: data for testing
    :param criterion: loss function
    :return: loss value and accuracy
    """
    test_loss = 0
    correct = 0
    model.eval()
    for data, target in test_data:
        output = model(data)
        test_loss += criterion(output, target).item()
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).sum().item()

    # loss function averages over batch size
    test_loss /= len(test_data)
    test_loss = format(test_loss, '.4f')
    acc = format(correct / len(test_data.dataset), '.4%')
    print('Test set: Average loss: {}, Accuracy: {}/{} ({})'
          .format(test_loss, correct, len(test_data.dataset), acc))
    return test_loss, acc


def predict(model, test_data, labels, file):
    """
    Make predictions about test sets.
    :param model: model for testing
    :param test_data: test data
    :param labels: correspondence between category numbers and labels
    :param file: the file to be written
    :return: void
    """
    for i, data in enumerate(test_data):
        imgs, names = data
        outputs = model(imgs)
        preds = outputs.data.max(1)[1]
        preds_list = preds.tolist()

        with open(file, 'a', encoding='utf-8') as f:
            for pred, name in zip(preds_list, names):
                f.write('{} {}\n'.format(name, labels[int(pred)]))


if __name__ == '__main__':

    # transform
    transform_data = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
    )


    # AlexNet Model
    model = AlexNet(num_classes=15)

    # prepare data
    trainset = datasets.ImageFolder(root=train_dir, transform=transform_data)
    trainloader = DataLoader(trainset, batch_size=10, shuffle=True)

    validateset = datasets.ImageFolder(root=validation_dir, transform=transform_data)
    validateloader = DataLoader(validateset, batch_size=10, shuffle=False)

    testset = TestImgData(test_dir=test_dir, transform=transform_data)
    testloader = DataLoader(testset, batch_size=10, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001,
                          momentum=0.9, weight_decay=1e-4)

    result_file = './run3usingAlexNet.txt'
    train = True

    if train:  # training process

        n_epochs = 50
        log = 10
        y_losses = []
        x_axis = []
        for epoch in range(1, n_epochs + 1):

            running_loss = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i % log == (log - 1):
                    print('[%d, %5d] loss: %.4f' %
                          (epoch, i + 1, running_loss / log))
                    y_losses.append(running_loss / log)
                    if len(x_axis) == 0:
                        x_axis.append(log)
                    else:
                        x_axis.append(x_axis[-1] + log)
                    running_loss = 0.0

            validate_model(model, validateloader, criterion=criterion)

        plt.plot(x_axis, y_losses)
        timestamp = int(time.time())
        plt.savefig('./train_loss_{}.png'.format(timestamp))

        # save model for testing usage
        torch.save(model, './model_{}.pkl'.format(timestamp))

    else:  # predict process
        model_path = './model_1542641124.pkl'
        model = torch.load(model_path)

        labels = get_labels(train_dir, imageFolder=trainset)
        predict(model, testloader, labels, result_file)

import torch.utils.data as data_utils
import torch
import pandas as pd
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
import itertools
from dotenv import load_dotenv, find_dotenv
import pickle
from wikiart import WikiartDataset
import os

load_dotenv(find_dotenv())


def get_classes(filepath):
    data = pd.read_csv(filepath, sep=',')
    return list(data['Genre'].unique()), list(data['Class'].unique())

genres, classes = get_classes(os.getenv('mappings'))

class _classifier(nn.Module):
    def __init__(self, pretrained_model, n_classes):
        super(_classifier, self).__init__()
        self.features = nn.Sequential(*list(pretrained_model.children())[:-1])
        self.classifier = nn.Sequential(nn.Linear(512, n_classes))

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y


def save_checkpoint(state, path='models/', filename='resnet18_re_checkpoint.pth.tar'):
    torch.save(state, path+filename)


def load_model(path):
    checkpoint = torch.load(path)
    net = checkpoint['model']
    state = checkpoint['state_dict']
    return net, state


def main(learning_rate, epochs=20):
    print("Loading training data....")

    train_path = os.getenv('train_path')
    test_path = os.getenv('test_path')
    train_images_path = os.getenv('train_aws_dataset')
    test_images_path = os.getenv('test_aws_dataset')
    bs = int(os.getenv('bs'))
    train_size = 1000
    test_size = 1000
    pickle_path = os.getenv('pickle_path')
    n_classes = int(os.getenv('genre_count'))

    print("Train path:", train_path)
    print("Test path:", test_path)
    print("Train images path:", train_images_path)
    print("Test images path", test_images_path)

    print("Loading train data....")
    wiki_train = WikiartDataset(config={'wikiart_path': train_path, 'images_path': train_images_path, 'size': train_size,
                                        'arch': 'resnet', 'train': True})

    print("Loading test data....")
    wiki_test = WikiartDataset(config={'wikiart_path': test_path, 'images_path': test_images_path, 'size': test_size,
                                       'arch': 'resnet', 'train': False})

    wiki_train_dataloader = data_utils.DataLoader(wiki_train, batch_size=bs, shuffle=True, num_workers=4,
                                                  drop_last=False)
    wiki_test_dataloader = data_utils.DataLoader(wiki_test, batch_size=bs, shuffle=True, num_workers=4,
                                                 drop_last=False)

    # net = _classifier(pretrained_model=resnet18(pretrained=True), n_classes=15)

    pretrained_model = resnet18(pretrained=True)

    net = _classifier(pretrained_model=pretrained_model, n_classes=10)

    # Defining the loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    # Train

    # for epoch in range(epochs):
    #     print("Epoch: ", epoch)
    #     running_loss = 0.0
    #
    #     for i, data in enumerate(wiki_train_dataloader, 0):
    #         inputs, labels = data['image'], data['class']
    #         batchsize = inputs.shape[0]
    #         # make this 4, 32, 32, 3 -> 4, 3, 32, 32
    #         inputs = inputs.view(batchsize, 3, 224, 224)
    #
    #         inputs, labels = Variable(inputs), Variable(labels)
    #         optimizer.zero_grad()
    #
    #         # forward + backward + optimize
    #         outputs = net(inputs)
    #
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         # print statistics
    #         running_loss += loss.data[0]
    #         if i % 50 == 49:  # print every 50 mini-batches
    #             print('[%d, %5d] loss: %.3f' %
    #                   (epoch + 1, i + 1, running_loss / 2000))
    #             running_loss = 0.0
    #
    # print('Finished Training')
    # save_checkpoint({'epoch': epochs, 'arch': 'resnet18_re', 'state_dict': net.state_dict(), 'model': net})

    net, state = load_model('models/resnet18_re_checkpoint.pth.tar')

    print("Predicting on the test set... ")
    class_correct = [i for i in range(n_classes)]
    class_total = [100] * n_classes

    y_pred = []
    y_actual = []

    for data in wiki_test_dataloader:
        images, labels = data['image'], data['class']
        batchsize = images.shape[0]
        images = images.view(batchsize, 3, 224, 224)
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()

        for i in range(batchsize):
            label = labels[0]
            y_actual.append(label)
            class_correct[label] += c[i]
            y_pred.append(c[i])

    print("Correct classes", class_correct)
    print("Total count for each class", class_total)

    print("Pickling predictions and labels")
    pkl1_name = "rnet18_re_ypred_{}_{}.pkl".format(learning_rate, epochs)
    pkl2_name = "rnet18_re_y_actual_{}_{}.pkl".format(learning_rate, epochs)

    with open(pickle_path + pkl1_name, 'wb') as f:
        pickle.dump(y_pred, f)

    with open(pickle_path + pkl2_name, 'wb') as f2:
        pickle.dump(y_actual, f2)

    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (
            genres[i], float(class_correct[i].item() * 100)/ class_total[i]))

if __name__ == '__main__':

    lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    epochs = [100]

    combinations = list(itertools.product(lrs, epochs))

    # for c in combinations:
    #     print("Learning rate {}, no of epochs {}".format(c[0], c[1]))
    #     main(learning_rate=c[0], epochs=c[1])

    main(learning_rate=0.001, epochs=20)
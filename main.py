import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets
from classifier import ResidualBlock, ResNet18
from torch.utils.data import DataLoader
from advdataset import AdvDataset
import time
from attack import fgsm_attack
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCHSIZE=256

train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(100),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(0, 360),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

data_transform = transforms.Compose([
        # transforms.RandomResizedCrop(100),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
    ])

train_dataset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=2)

test_dataset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=data_transform)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def evaluate(model, data_loader, criterion):
    avg_loss = 0.0
    total = 0.
    accuracy = 0.
    test_loss = []
    model.eval()
    for batch_num, (feats, labels) in enumerate(data_loader):
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats)
        
        prob = F.softmax(outputs.detach(), dim=1)
        
        _, pred_labels = torch.max(prob, 1)
        pred_labels = pred_labels.view(-1)

        loss = criterion(outputs, labels.long())
        
        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()]*feats.size()[0])

        torch.cuda.empty_cache()
        del outputs
        del feats
        del labels
        del pred_labels
        del prob

    return np.mean(test_loss), accuracy/total

criterion = nn.CrossEntropyLoss()

model = ResNet18(ResidualBlock).to(device)

# model = DenseNet121().to(device)
optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.01, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

def train(model, train_loader, optimizer, criterion,  epochs):
    model.train()
    start_time = time.time()

    for epoch in range(epochs):
        total = 0.0
        train_acc = 0.0
        train_loss = []
        total_labels = []

        model.train()
        for batch_num, (feats, labels) in enumerate(train_loader):
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(feats)
                loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            prob = F.softmax(outputs.detach(), dim=1)

            _, pred_labels = torch.max(prob, 1)
            pred_labels = pred_labels.view(-1)
            train_acc += torch.sum(torch.eq(pred_labels, labels)).item()    
            
            total += len(labels)
            train_loss.extend([loss.item()]*feats.size()[0])
            torch.cuda.empty_cache()

            del loss
            del feats
            del labels
            del pred_labels
            del outputs
            del prob

        scheduler.step()

        torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    #'scheduler_state_dict' : scheduler.state_dict(),
        }, "Model_"+str(epoch))
        val_loss, val_acc = evaluate(model, test_loader, criterion)
        print('epoch: %d\t'%(epoch+1),  'time: %d m: %d s '% divmod(time.time() - start_time, 60))
        start_time = time.time()
        print('train_loss: %.5f\ttrain_acc: %.5f' %(np.mean(train_loss), train_acc/total))
        print('val_loss: %.5f\tval_acc: %.5f'% (val_loss, val_acc))
        print('*'*60)
        
def test( model, device, test_loader, test_dataset, epsilon ):
    
    # Accuracy counter
    correct = 0
    adv_examples = []
    pred_list = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # mask: 1 for correct, only update grad on correct image
        mask = torch.eq(init_pred.flatten(), target.flatten()).float()

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad, mask)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # calculate correct prediction
        correct += torch.sum(torch.eq(final_pred.flatten(), target.flatten())).item()
        # Special case for saving 0 epsilon examples
        
        adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
        adv_examples.append( (target.flatten().detach().cpu().numpy(), adv_ex) )
        pred_list.append((init_pred.flatten().detach().cpu().numpy(), final_pred.flatten().detach().cpu().numpy()))

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_dataset))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_dataset), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples, pred_list


train_loss_list = []
train_acc1_list = []
val_acc_list = []
val_loss_list = []
train_acc2_list = []
def adv_train(model, train_loader, test_loader, optimizer, criterion, epsilon, epochs):
  model.train()
  start_time = time.time()

  for epoch in range(epochs):
    total = 0.0
    train_acc1 = 0.0
    train_acc2 = 0.0
    train_loss = []
    total_labels = []

    model.train()
    for batch_num, (feats, labels) in enumerate(train_loader):

      feats, labels = feats.to(device), labels.to(device)

      # Set requires_grad attribute of tensor. Important for Attack
      feats.requires_grad = True

      optimizer.zero_grad()

      # calculate grad on origional image
      # with torch.cuda.amp.autocast():
      outputs1 = model(feats)
      loss1 = criterion(outputs1, labels.long())

      # create a copy to backward
      temp_feats = copy.deepcopy(feats)
      temp_model = copy.deepcopy(model)
      temp_outputs = temp_model(temp_feats)
      temp_loss = criterion(temp_outputs, labels.long())
      temp_model.zero_grad()
      temp_loss.backward()
      # generate adv image and calculate loss
      adv_ex = fgsm_attack(temp_feats, epsilon, temp_feats.grad.data, mask=torch.ones(feats.shape[0]).to(device))

      # with torch.cuda.amp.autocast():
      outputs2 = model(adv_ex)
      loss2 = criterion(outputs2, labels.long())

      # new loss,  update gradient loss on new loss
      loss = (loss1+loss2)/2
      loss.backward()
      
      optimizer.step()


      # calculate accuracy on original image
      prob1 = F.softmax(outputs1.detach(), dim=1)

      _, pred_labels1 = torch.max(prob1, 1)
      pred_labels1 = pred_labels1.view(-1)
      train_acc1 += torch.sum(torch.eq(pred_labels1, labels)).item()    

      # calculate accuracy on attack image
      prob2 = F.softmax(outputs2.detach(), dim=1)

      _, pred_labels2 = torch.max(prob2, 1)
      pred_labels2 = pred_labels2.view(-1)
      train_acc2 += torch.sum(torch.eq(pred_labels2, labels)).item()    
      
      total += len(labels)
      train_loss.extend([loss.item()]*feats.size()[0])
      torch.cuda.empty_cache()

      del loss
      del feats
      del labels
      del pred_labels1
      del pred_labels2
      del outputs1
      del outputs2
      del loss1
      del loss2
      del prob1
      del prob2
      del temp_feats
      del temp_loss
      del temp_outputs
      del temp_model
      del adv_ex
    #scheduler.step()

    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                #'scheduler_state_dict' : scheduler.state_dict(),
    }, "Model_"+str(epoch))
    val_loss, val_acc = evaluate(model, test_loader, criterion)
    print('epoch: %d\t'%(epoch+1),  'time: %d m: %d s '% divmod(time.time() - start_time, 60))
    start_time = time.time()
    train_loss_list.append(np.mean(train_loss))
    train_acc1_list.append(train_acc1/total)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)
    train_acc2_list.append(train_acc2/total)


    print('train_loss: %.5f\ttrain_acc1: %.5f' %(np.mean(train_loss), train_acc1/total))
    print('val_loss: %.5f\tval_acc: %.5f'% (val_loss, val_acc))
    print('attack_acc: %.5f'%(train_acc2/total))
    print('*'*70)

eps = 0.05
acc,ex, pred = test(model, device, test_loader, test_dataset, eps)

label = []
adv_ex = []

label = [j for i in ex for j in i[0]]
adv_ex = [j for i in ex for j in i[1]]

advdataset = AdvDataset(adv_ex, label)
advloader = DataLoader(advdataset, BATCHSIZE, shuffle=True)

advModel = ResNet18(ResidualBlock, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(advModel.parameters(), momentum=0.9, lr=0.01, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

adv_train(advModel, train_loader, advloader, optimizer,criterion, 0.05, 20)


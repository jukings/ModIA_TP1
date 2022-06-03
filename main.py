import argparse
from mnist_net import MNISTNet
from torch import optim

if __name__=='__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument('--exp_name', type=str, default = 'MNIST', help='experiment name')
  parser.add_argument('--batch_size', type=int, default = 4, help='batch size')
  parser.add_argument('--epochs', type=int, default = 16, help='epochs')
  parser.add_argument('--lr', type=float, default = 10**-4, help='learning rate')

  args = parser.parse_args()
 
  # transforms
  transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))])

  # datasets
  trainset = torchvision.datasets.MNIST('./data', download=True, train=True, transform=transform)
  testset = torchvision.datasets.MNIST('./data', download=True, train=False, transform=transform)

  # dataloaders
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

  net = MNISTNet()
  # setting net on device(GPU if available, else CPU)
  net = net.to(device)
  optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

  train(...)
  test_acc = test(...)
  print(f'Test accuracy:{test_acc}')
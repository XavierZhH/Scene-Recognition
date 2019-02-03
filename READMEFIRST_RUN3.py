#To run this codes, pytorch should be installed 

#Instrcution: 
#   There are three parts in our algorithm of 'AlexNet', they are alexnet, utils
#       and run3.

#   'alexnet' is the training model.

#   'utils' is doing some image preprocessings, including wrap the bow features 
#           into data that pytorch can traverse, read test images and get 
#           their file names to write text, randomly generate validation data 
#           from the training data, get test images, get the correspondence
#           between category numbers and tag names, and zero mean and unit 
#           length conversion.

#   ‘run3’ is the main program, it includes both parts above, and test model 
#           classification accuracy, make predictions about test sets and 
#           storing the result as a txt file.



#Usage:
#      'utils.py':
#           BoWData(Dataset):
#                Wrap the bow features into data that pytorch can traverse.
#           TestImgData(Dataset):
#               Read test images and get their file names to write text.
#           generate_validate_data(validation_dir, train_dir, ratio=0.2):
#               param validation_dir: the path where validation data is stored
#               param train_dir: the path where training data is stored
#               param ratio: generation ratio
#               return: void
#           get_test_data(test_dir, transform=None):
#               param test_dir: the path where test data is stored
#               param transform: transform function
#               return: test images information list
#           get_labels(dir, imageFolder=None):
#               param dir: the path where training data is stored
#               param imageFolder: pytorch data structure
#               return: the correspondence between category numbers and tag names
#           zero_mean_and_unit_length(tensor):
#               param tensor: target tensor
#               return: transformed tensor

#       'run3,py':
#           validate_model(model, test_data, criterion=torch.nn.NLLLoss()):
#               param model: the model for testing
#               param test_data: data for testing
#               param criterion: loss function
#               return: loss value and accuracy
#           predict(model, test_data, labels, file):
#               param model: model for testing
#               param test_data: test data
#               param labels: correspondence between category numbers and labels
#               param file: the file to be written
#               return: void
#           



#Written by Hang Zhong and Haojiong Wang.
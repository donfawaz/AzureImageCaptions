import sys
sys.path.append('./cocoapi/PythonAPI')

import torch
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as data
from azureml.core import Run

from pycocotools.coco import COCO
from data_loader import get_loader
from model import EncoderCNN, DecoderRNN

import numpy as np
import os
import requests
import time
import math
import argparse



# Get the Azure ML object
run = Run.get_context()

# Training setting 
parser = argparse.ArgumentParser(description='PyTorch image captions on coco dataset')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training')
parser.add_argument('--vocab_threshold', type=int, default=3, help='minimum word count threshold')
parser.add_argument('--vocab_from_file', type=bool, default=True, help='if true, load existing vocab file')
parser.add_argument('--embed_size', type=int, default=300, help='dimensionality of image and word embeddings')
parser.add_argument('--hidden_size', type=int, default=512, help='number of features in hidden state of the RNN decoder')
parser.add_argument('--num_epochs', type=int, default=20, help='number of training epoch')
parser.add_argument('--save_every', type=int, default=1, help='determines frequency of saving model weights')
parser.add_argument('--print_every', type=int, default=100, help='determines window for printing average loss')
parser.add_argument('--data_folder', type=str, help='path to dataset folder')
args = parser.parse_args()

# Image transformations
transform_train = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])


# Build data loader.
data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=args.batch_size,
                         vocab_threshold=args.vocab_threshold,
                         vocab_from_file=args.vocab_from_file,
                         cocoapi_loc=args.data_folder)


# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)

# Initialize the encoder and decoder. 
encoder = EncoderCNN(args.embed_size)
decoder = DecoderRNN(args.embed_size, args.hidden_size, vocab_size)

# Move models to GPU if CUDA is available. 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
decoder.to(device)

# Define the loss function. 
criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

#3: Specify the learnable parameters of the model.
params = list(decoder.parameters()) + list(encoder.embed.parameters())

#4: Define the optimizer.
optimizer = torch.optim.Adam(params, lr=0.0001)

# Set the total number of training steps per epoch.
total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)


for epoch in range(1, args.num_epochs+1):
    
    for i_step in range(1, total_step+1):
        
        # Randomly sample a caption length, and sample indices with that length.
        indices = data_loader.dataset.get_train_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader.batch_sampler.sampler = new_sampler
        
        # Obtain the batch.
        images, captions = next(iter(data_loader))

        # Move batch of images and captions to GPU if CUDA is available.
        images = images.to(device)
        captions = captions.to(device)
        
        # Zero the gradients.
        decoder.zero_grad()
        encoder.zero_grad()
        
        # Pass the inputs through the CNN-RNN model.
        features = encoder(images)
        outputs = decoder(features, captions)
        
        # Calculate the batch loss.
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        
        # Backward pass.
        loss.backward()
        
        # Update the parameters in the optimizer.
        optimizer.step()
            
        # Get training statistics.
        stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (epoch, args.num_epochs, i_step, total_step, loss.item(), np.exp(loss.item()))
        
        # Log the loss to Azure ML
        run.log('loss', loss.item())
        run.log('perplexity', np.exp(loss.item()))
        run.log('stats', stats)

        # Print training statistics (on same line).
        print('\r' + stats, end="")
        sys.stdout.flush()
        
        
        # Print training statistics (on different line).
        if i_step % args.print_every == 0:
            print('\r' + stats)
        
    # Save the weights.
    if epoch % args.save_every == 0:
        torch.save(decoder.state_dict(), os.path.join('./models', 'decoder-%d.pkl' % epoch))
        torch.save(encoder.state_dict(), os.path.join('./models', 'encoder-%d.pkl' % epoch))

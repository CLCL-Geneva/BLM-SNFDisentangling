
"""

"""


import argparse

import json

import numpy as np
import random
import torch

import utils.embeddings as embeddings
import utils.data_extractor as data_extractor
import baselines.train as train
import baselines.test as test
import baselines.baseline


from datetime import datetime

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="/home/vivi/work/Projects/BLMs/blm_data", help="data directory")

    parser.add_argument("--sent_emb_dim", default=768, help="Dimension of sentence embedding")
    parser.add_argument("--transf", default="bert", choices=["bert", "flaubert"], help="which embeddings to use")
    
    parser.add_argument("--epochs", "-e", default = 100, help="Number of epochs.")
    parser.add_argument("--lr", "-l", default=1e-2, help="Learning rate.")
    parser.add_argument("--batch_size", "-bs", default=100, help="batch size")
    parser.add_argument("--seq_size", "-s", type=int, default=7, help="Size of the matrix for VAE initialisation.")
    parser.add_argument("--model", "-v", help="path to the saved model parameters.")
    parser.add_argument("--extract", "-r", action="store_true", help="If extract, it extract ONLY bert embeddings from sentences.")
    parser.add_argument("--dict", "-d", action="store_true", help="If dict, it creates a dictionary of words for reconstruction.")
    parser.add_argument("--train", "-t", action="store_true", help="Load embeddings from file and compute training.")
    parser.add_argument("--test", action="store_true", help="If true, do only the evaluation step.")
    parser.add_argument("--cuda", action="store_true", help="Enable GPU")
    parser.add_argument("--baseline", "-ba", choices=["FFNN", "CNN"], default="FFNN", help="Type of baseline (CNN|FFNN)")
    parser.add_argument("--type", "-ty", choices=["type_I", "type_II", "type_III"], default="type_I", help="Process type I, II or III.")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%d-%b-%Y_%H:%M")

    data_path = args.data_dir
    input_train = args.data_dir + "/" + args.type + "/datasets/train/sentences/"
    input_test = args.data_dir + "/" + args.type + "/datasets/test/sentences/"
    output_dir = args.data_dir + "/" + args.type + "/output" 
    model_path = output_dir + "/baseline_" + args.baseline + "_" + str(args.epochs) + "epochs_" + str(args.lr) + "lr_" + timestamp + "/"


    # setting the seed for reproducibility
    print("Setting seeds...")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    '''
    # enable cuda
    args.device = None
    args.cuda=True
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        print("cuda available...")
    else:
        args.device = torch.device("cpu")
        print("cuda not available...")
    '''
    args.device = torch.device("cpu")


    train_dir = data_path + "/" + args.type + "/datasets/train/embeddings/" + args.transf + "_sentembs/"
    test_dir = data_path + "/" + args.type + "/datasets/test/embeddings/" + args.transf + "_sentembs/"

    # vectors for training
    # USE BATCHES TO AVOID OUT OF MEMORY!!!
    args.extract = False
    if args.extract:
        # print("Extracting training embeddings...")
        embeddings.extract_emb(input_train, train_dir, args)
        print("Extracting test embeddings...")
        embeddings.extract_emb(input_test, test_dir, args)
        


    args.seq_size = 7
    
    args.train = True
    
    #initialize the model -- for training and testing
    print("Model initialitation...")
    if "cnn" in args.baseline.lower():
        model = baselines.baseline.BaselineCNN(args.sent_emb_dim, int(args.seq_size))
    elif "ffnn" in args.baseline.lower():
        model = baselines.baseline.BaselineFFNN(args.sent_emb_dim, int(args.seq_size))
    else:
        print("Unknown model. defaulting to FFNN")
        model = baselines.baseline.BaselineFFNN(args.sent_emb_dim, int(args.seq_size))


    if args.train:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=float(args.lr))

        train_x, train_y, train_truth = embeddings.load_embeddings(train_dir)

        print("Training data: \n\tx => {}\n\ty => {},\n\ttruth => {}".format(train_x.size(), train_y.size(), train_truth.size()))

        nr_train_insts = int(len(train_x) * 0.8)

        # Validation data
        valid_x = train_x[nr_train_insts:]
        valid_y = train_y[nr_train_insts:]
        valid_t = train_truth[nr_train_insts:]

        #training data
        train_x = train_x[:nr_train_insts]
        train_y = train_y[:nr_train_insts]
        train_truth = train_truth[:nr_train_insts]
                
        train_size = train_x.shape[0]

        trainloader = data_extractor.data_loader(train_x, train_y, train_truth, int(args.batch_size), True)
        validloader = data_extractor.data_loader(valid_x, valid_y, valid_t, int(args.batch_size), True)
        print(len(trainloader), len(validloader))

        print("Training...")
        train_loss_list, valid_loss_list = train.training(trainloader, validloader, model, int(args.epochs), optimizer, train_size, args.test, float(args.lr), args.device, args.type, model_path)

    args.test = True

    if args.test:
        batch_size = 1
        print("Preparing the data for testing...")
        print("\tBERT embeddings...")
        
        test_x, test_y, test_truth = embeddings.load_embeddings(test_dir)
        
        print("Test data: \n\tx => {}\n\ty => {},\n\ttruth => {}".format(test_x.size(), test_y.size(), test_truth.size()))

        with open(input_test + "/labels.json", "r", encoding="utf-8") as file:
            labels = json.load(file)
        with open(input_test + "/type_file.json", "r", encoding="utf-8") as file:
            type_file = json.load(file)
            
        test_size = test_x.shape[0]
        print("\tDataloader...")
        testloader = data_extractor.data_loader(test_x, test_y, test_truth, batch_size, False)

        # Load and test all the saved models
        print("Loading model from {} ...".format(model_path + "/best_model.pth"))
        
        try:
            model.load_state_dict(torch.load(model_path + "/best_model.pth", map_location=args.device))
        except RuntimeError as e:
            print("ERROR: --> ", model_path + "/best_model.pth")

        print("Testing...")
        test.testing(testloader, input_test, model_path, labels, type_file, model, test_size, args.test, args.device, args.type)


if __name__ == '__main__':
    main()


"""

"""


import argparse

import os

import json

import torch

from utils import embeddings, data_extractor
import baselines.test as test
import baselines.baseline


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="/home/vivi/work/Projects/BLMs/blm_data/", help="data directory")

    parser.add_argument("--test_type", help="data type for testing")
    parser.add_argument("--output_dir", "-o", help="Output directory.")
    
    parser.add_argument("--sent_emb_dim", default=768, help="Dimension of sentence embedding")
    
    parser.add_argument("--model", help="path to the saved model parameters.")

    args = parser.parse_args()

    data_dir = args.data_dir

    input_test = data_dir + "/" + args.test_type +  "/datasets/test/sentences/"
    test_dir = data_dir + "/" + args.test_type + "/datasets/test/embeddings/"
    test_type = args.test_type

    output_dir = args.output_dir
    os.makedirs(output_dir,exist_ok=True)

    model_file = args.model
    
    results_path = output_dir + "/" + model_file.split("/")[-2] + "_test-on_" + test_type + "/"

    # enable cuda
    args.device = None
    args.cuda=True
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        print("cuda available...")
    else:
        args.device = torch.device("cpu")
        print("cuda not available...")

    #initialize the model -- for training and testing
    args.seq_size = 7
    print("Model initialization...")
    if "cnn" in model_file.lower():
        model = baselines.baseline.BaselineCNN(args.sent_emb_dim, int(args.seq_size))
    elif "ffnn" in model_file.lower():
        model = baselines.baseline.BaselineFFNN(args.sent_emb_dim, int(args.seq_size))
    else:
        print("Unknown model. defaulting to FFNN")
        model = baselines.baseline.BaselineFFNN(args.sent_emb_dim, int(args.seq_size))


    args.seq_size = 7

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
        print("Loading model from {} ...".format(model_file))
        
        try:
            model.load_state_dict(torch.load(model_file, map_location=args.device))
        except RuntimeError as e:
            print("ERROR {}: --> {}".format(e,model_file))

        print("Testing...")
        test.testing(testloader, input_test, results_path, labels, type_file, model, test_size, args.test, args.device, test_type)


if __name__ == '__main__':
    main()

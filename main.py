import os
import numpy as np

# Set Global variables
import src.glo as glo
glo._init()
glo.set_value('EPS', 1e-16)
glo.set_value('LRP_FILTER_TOP_K', 0.1)

from src.utils import *
from src.scRNAdata import *
from src.train import *
from src.plot import *
from src.lrp import *

from src.model import protoCloud



def main(args):
    ### load dataset
    data = scRNAData(args)
    # ordered_celltype = ordered_class(data, args)
    if data.Y is not None:
        data_info_saver(data.cell_encoder, args, 'cell_encoder')

    if args.pretrain_model_pth is not None:
        print("Reformat data according to pretrained model")
        # resize data to model input dim
        data.gene_subset(args.pretrain_model_pth, args)
        data_info_saver(data.gene_names, args, 'gene_names')
    else:
        args.model_data = None
    

    print('\nData: ', args.dataset)
    # set dataloader
    if args.model_mode in ['train', 'plot']:
        (train_X, test_X, train_Y, test_Y) = data.split_data(args)
        train_loader = data.assign_dataloader(train_X, train_Y, args.batch_size)
        print('Training set size: {0}'.format(train_X.shape[0]))

    elif args.model_mode  in ['test', 'apply']:
        test_X = data.X
        test_Y = data.Y
        args.training_size = 0
    test_X = torch.Tensor(test_X)
    test_Y = torch.LongTensor(test_Y) if test_Y is not None else test_Y
    print('Test set size: {0}'.format(test_X.shape[0]))


    # Setup model
    #######################################################
    if args.pretrain_model_pth is None:
        # model settings based on data
        args.input_dim = len(data.gene_names)
        args.num_classes = len(data.cell_encoder.classes_)
        args.num_prototypes = args.num_classes * args.prototypes_per_class
        print('Num genes: {0}'.format(args.input_dim))
        print('Num classes: {0}'.format(args.num_classes))

        # setup model
        print('\nSetup model')
        model_dict = {"raw_input": args.raw,
                    "input_dim": args.input_dim,
                    "latent_dim": args.latent_dim,
                    "num_prototypes": args.num_prototypes,
                    "num_classes": args.num_classes,
                    "activation": args.activation,
                    "use_bn": args.use_bn,
                    "obs_dist": args.obs_dist,
                    "nb_dispersion": args.nb_dispersion,
                    "use_bias": args.use_bias,
                    "use_dropout": args.use_dropout,
                    "encoder_layer_sizes": args.encoder_layer_sizes,
                    "decoder_layer_sizes": args.decoder_layer_sizes,
                    # "n_batch": args.n_batch,
                    }
        print(model_dict)
        model = protoCloud(**model_dict).to(device)
        # print(model)
        if args.cont_train or args.model_mode != "train":
            print('\nLoading existing model')
            load_model(args.model_dir + args.exp_code + '.pth', model)
    
    else:   # args.pretrain_model_pth not None
        print('\nLoad existing model from: \n\t', args.pretrain_model_pth)
        model_dict = load_model_dict(os.path.dirname(args.pretrain_model_pth))
        print("Loaded model parameters: \n\t", model_dict)

        args.input_dim = model_dict["input_dim"]
        args.num_classes = model_dict["num_classes"]
        args.num_prototypes = model_dict["num_prototypes"]
        args.prototypes_per_class = model_dict["num_prototypes"] // model_dict["num_classes"]
        
        model = protoCloud(**model_dict).to(device)
        load_model(args.pretrain_model_pth, model)

        # copy pretrained model label encoder
        if args.cont_train:
            model_encoder = data_info_loader('cell_encoder', os.path.dirname(args.pretrain_model_pth))
            data_info_saver(model_encoder, args, 'cell_encoder')

    

    # Training
    #######################################################
    if args.model_mode == "train":
        print('\nEnter training')
        result_trend = run_model(model, train_loader, test_X, test_Y, 
                                args.epochs, args.lr, args.optimizer,
                                args.two_step, args.ortho_coef,
                                args.early_stopping)

        # save model to model_dir
        save_model(model, args.model_dir, args.exp_code)
        save_model_dict(model_dict, args.model_dir)
        save_file(result_trend, args, '_trend.npy')  # save loss & accuracy through epochs
        save_file(get_prototypes(model), args, '_prototypes.npy') # save prototypes
        # save model's celltype & gene names
        data_info_saver(data.gene_names, args, 'gene_names')
        
        print('\tTraining information saved')
    
    else:
        args.plot_trend = 0
        


    # Predictions
    #######################################################
    if args.model_mode != 'plot':
        predicted, _ = get_predictions(model, test_X, False if args.model_mode == 'apply' else True)
        # predicted['celltype'] = test_Y
        if args.pretrain_model_pth is not None:
            model_encoder = data_info_loader('cell_encoder', os.path.dirname(args.pretrain_model_pth))
            predicted['idx1'] = model_encoder.inverse_transform(predicted['idx1'])
            predicted['idx2'] = model_encoder.inverse_transform(predicted['idx2'])
        else:
            predicted['idx1'] = data.cell_encoder.inverse_transform(predicted['idx1'])
            predicted['idx2'] = data.cell_encoder.inverse_transform(predicted['idx2'])
        
        if args.new_label: # save orig label as actual label
            test_idx = load_file(args, '_idx.csv')['test_idx'].dropna().astype(int)
            predicted['celltype'] = data.adata.obs["celltype"][test_idx]
        elif test_Y is None:
            pass
        else:
            predicted['celltype'] = data.cell_encoder.inverse_transform(test_Y)

        predicted = pd.DataFrame(predicted)
        save_file(predicted, args, '_pred.csv')
        print("\nPredictions saved")

        if model.obs_dist == 'nb' and test_Y is not None:
            # log-likelihood
            ll = model.get_log_likelihood(torch.tensor(test_X[:1000]).to(device), 
                                        torch.tensor(test_Y[:1000]).to(device)).detach().cpu().numpy()
            save_file(ll, args, '_ll.npy')
            print("Avg log-likelihood: ", np.mean(ll))
            plot_cell_likelihoods(args, ll)

        # misclass_rate, rep, cm
        results = model_metrics(predicted)
        save_file(results, args, '_metrics.npy')

        latent = get_latent(model, test_X)
        save_file(latent, args, '_latent.npy')
        proto = get_prototypes(model)
        save_file(proto, args, '_prototypes.npy')
            


    # Visualization
    #######################################################
    print('Visualizing results')
    if args.plot_trend:
        plot_epoch_trend(args)
    if args.cm and test_Y is not None:
        plot_confusion_matrix(args)
    if args.umap:
        plot_umap_embedding(args, data)
    if args.two_latent:
        plot_two_latents_embedding(args, data)
    if args.protocorr:
        plot_protocorr_heatmap(args, data)
    if args.distance_dist:
        plot_distance_to_prototypes(args, data)


    #######################################################
    # LRP based explanations & marker gene selection
    if args.prp or args.lrp:
        print("\nGenerating model explanations")
        wrapper = model_canonized()
        # construct the model for generating LRP based explanations
        model_wrapped = protoCloud(**model_dict).to(device)
        # replace layers in model_wrapped with customized LRP layers
        wrapper.copyfrommodel(model_wrapped, 
                              model, 
                              lrp_params=lrp_params_def1,
                              lrp_layer2method=lrp_layer2method)
        print("\tModel wrapped")

        if args.prp:
            generate_PRP_explanations(model_wrapped, 
                                model.prototype_vectors, 
                                test_X, test_Y, 
                                data,
                                model.epsilon, 
                                args,
                                )
        if args.lrp:
            #TODO: misclassified genes LRP
            generate_LRP_explanations(model_wrapped, 
                                test_X, test_Y,
                                data,
                                model.epsilon, 
                                args,
                                )
    
    if args.plot_prp:
        print("Ploting PRP visulization")
        plot_prp_dist(data, args)
    if args.plot_lrp:
        print("Ploting LRP visulization")
        plot_lrp_dist(data, args)
        plot_top_gene_heatmap(data, args)
        plot_outlier_heatmap(data, args)
        plot_marker_venn_diagram(data.adata, args)
    

    print("Finished!")
        





import argparse

parser = argparse.ArgumentParser(description = 'ProtoCloud')
#######################################################
### ----Path Parameters----
parser.add_argument('--data_dir',       type = str, default = './data/')
parser.add_argument('--model_dir',      type = str, default = './saved_models/')
parser.add_argument('--results_dir',    type = str, default = './results/')
### ----Data Parameters----
parser.add_argument('--dataset', type = str, default = 'PBMC_10K',
                    #  choices = ['PBMC_10K', 'PBMC_20K', 'TSCA_lung', 'TSCA_spleen', 'TSCA_oesophagus', 'RGC', 'GCA', 'SM3'],
                     )
parser.add_argument('--raw',        type = int, default = 1, choices = [0, 1], help = 'use raw data or normalized data')
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--topngene',   type = int, help = 'number of genes to select')
parser.add_argument('--test_ratio', type = float, default = 0.1)
parser.add_argument('--data_balance', type = int, default = 1, choices = [0, 1], help = 'use weighted sampling for training data or not')
parser.add_argument('--split',      type = str, default = None,
                    help = "cluster name of test data. if None, use test_ratio")
parser.add_argument('--new_label',  type = int, default = 0, choices = [0, 1], help = 'use previous predicted label as target or not')

#######################################################
### ----Model Parameters----
parser.add_argument('--model',       type = str, default = 'protoCloud', 
                    # choices = ['celltypist', 'svm', 'protoCloud', 'protoCloud0', 'protoCloud1', 'protoCloud2'],
                    )
parser.add_argument('--model_mode',  type = str, default = "test", choices = ["train", "test", "apply", "plot"],
                    help = "train model; test: z_mu for pred; apply: full test with reparametrization; plot: load and plot result files")
parser.add_argument('--cont_train',  type = int, default = 0, help = 'Load existing model and continue training')
parser.add_argument('--pretrain_model_pth',  type = str, help = 'Full path of pre-trained model to load')
parser.add_argument("--encoder_layer_sizes", nargs='+', type=int, help="List of values")
parser.add_argument("--decoder_layer_sizes", nargs='+', type=int, help="List of values")

# Optimizer Parameters + Survival Loss Function
parser.add_argument('--two_step',    type = int, default = 1, choices = [0, 1], help = 'use two-step training or not')
parser.add_argument('--early_stopping', type = int, default = 0, choices = [0, 1], help = 'use early stopping training or not')
parser.add_argument('--ortho_coef',  type = float, default = 0.3, help = 'orthogonality_loss coefficient')
parser.add_argument('--activation',  type = str, default = 'relu', choices = ['relu'])
parser.add_argument('--use_bias',      type = int, default = 0, choices = [0, 1])
parser.add_argument('--use_dropout',      type = float, default = 0.0)
parser.add_argument('--use_bn',      type = int, default = 1, choices = [0, 1], help = 'use batch normalization or not')
parser.add_argument('--optimizer',   type = str, default = 'AdamW', choices = ['Adam', 'AdamW'])
parser.add_argument('--lr',          type = float, default = 1e-3)
parser.add_argument('--epochs',      type = int, default = 150)
parser.add_argument('--target_accu', type = float, default = 0.7, help = 'mininum target accuracy to save model')
parser.add_argument('--seed',        type = int, default = 7)
# Loss Function Parameters
parser.add_argument('--obs_dist',       type = str, default = 'nb', choices = ['nb', 'normal'])
parser.add_argument('--nb_dispersion',  type = str, default = 'celltype_target', choices = ['celltype_target', 'celltype_pred', 'gene'])
# protoCloud Parameters
parser.add_argument('--prototypes_per_class',   type = int, default = 6)
parser.add_argument('--latent_dim',             type = int, default = 20)
#######################################################
### Result plotting Parameters
parser.add_argument('--visual_result',  type = int, default = 1, choices = [0, 1], help = 'Generate all visual results')
parser.add_argument('--plot_trend',     type = int, default = 0, choices = [0, 1], help = 'plot training trend vs epoch')
parser.add_argument('--cm',             type = int, default = 0, choices = [0, 1], help = 'plot confusion matrix')
parser.add_argument('--umap',           type = int, default = 0, choices = [0, 1], help = 'plot umap')
parser.add_argument('--two_latent',        type = int, default = 0, choices = [0, 1], help = 'plot misclassified points umap')
parser.add_argument('--protocorr',      type = int, default = 0, choices = [0, 1], help = 'plot prototype correlation')
parser.add_argument('--distance_dist',       type = int, default = 0, choices = [0, 1], help = 'plot latent distance distribution to prototypes')

parser.add_argument('--prp',            type = int, default = 1, choices = [0, 1], help = 'generate all PRP based explanations')
parser.add_argument('--lrp',            type = int, default = 1, choices = [0, 1], help = 'generate LRP based explanations')
parser.add_argument('--plot_prp',       type = int, default = 0, choices = [0, 1], help = 'plot all PRP explanation plots')
parser.add_argument('--plot_lrp',       type = int, default = 0, choices = [0, 1], help = 'plot LRP explanation plots')


args = parser.parse_args()


# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_workers = 4 if torch.cuda.is_available() else 0

# set random seed
seed_torch(torch.device(device), args.seed)

### Creates Experiment Code from argparse + Folder Name to Save Results
args.exp_code = get_custom_exp_code(args)
print("Experiment Name:", args.exp_code)


# data directory
makedir(args.data_dir)

# save model directory
model_dir = args.model_dir + args.dataset + '/'
args.model_dir = model_dir
makedir(args.model_dir)

# results directory
results_dir = args.results_dir + args.dataset + '/'
args.results_dir = results_dir
makedir(results_dir)
args.plot_dir = args.results_dir + 'plots/'
makedir(args.plot_dir )

args.prp_path = args.results_dir + 'prp/'
makedir(args.prp_path)
args.lrp_path = args.results_dir + 'lrp/'
makedir(args.lrp_path)


if args.cont_train:
    args.two_step = 0

if args.visual_result:
    args.plot_trend = 1
    args.cm = 1
    args.umap = 1
    args.two_latent = 1
    args.protocorr = 1
    args.distance_dist = 1

if args.lrp:
    args.plot_lrp = 1
if args.prp:
    args.lrp = 1
    args.plot_prp = 1
    args.plot_lrp = 1


print('------args---------')
print(args)



if __name__ == '__main__':
    main(args)
    print("end script\n\n\n\n\n\n")
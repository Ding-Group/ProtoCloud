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
    args_dict = vars(args)
    
    ### load dataset
    data = scRNAData(**args_dict)
    # ordered_celltype = ordered_class(data, args)

    if args.pretrain_model_pth is not None:
        print("Reformat data according to pretrained model")
        # resize data to model input dim
        data.gene_subset(args.pretrain_model_pth)
        data_info_saver(data.gene_names, args.model_dir, 'gene_names')
    

    print('\nData: ', args.dataset_name)
    # set dataloader, test_Y is not encoded
    if args.model_mode in ['train', 'test', 'plot']:
        train_idx, test_idx = data.get_split_idx(**args_dict)
        (train_X, test_X, train_Y, test_Y) = data.split_data(train_idx, test_idx, **args_dict)
        if train_X.shape[0] > 1e5 and args.batch_size == 128:
            args.batch_size = 1024
        train_loader = data.assign_dataloader(train_X, train_Y, args.batch_size)
        data_info_saver(data.cell_encoder, args.model_dir, 'cell_encoder')
        print('Training set size: {0}'.format(train_X.shape[0]))

    elif args.model_mode == 'apply':
        (_, test_X, _, test_Y) = data.split_data(**args_dict)
    test_X = torch.Tensor(test_X)
    print('Test set size: {0}'.format(test_X.shape[0]))
    args.model_validation = 0 if test_Y is None else args.model_validation


    # Setup model
    #######################################################
    if args.pretrain_model_pth is None:
        # model settings based on data
        args.input_dim = len(data.gene_names)
        args.num_classes = len(np.unique(train_Y))
        args.num_prototypes = args.num_classes * args.prototypes_per_class
        print('Num genes: {0}'.format(args.input_dim))
        print('Num classes: {0}'.format(args.num_classes))

        # setup model
        print('\nSetup model')
        model_dict = {"raw_input": args.raw,
                    "input_dim": args.input_dim,
                    "latent_dim": args.latent_dim,
                    "num_prototypes_per_class": args.prototypes_per_class,
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
        model_dict = load_model_dict(os.path.dirname(args.pretrain_model_pth), device)
        print("Loaded model parameters: \n\t", model_dict)

        args.input_dim = model_dict["input_dim"]
        args.num_classes = model_dict["num_classes"]
        args.prototypes_per_class = model_dict["num_prototypes_per_class"]
        args.num_prototypes = args.prototypes_per_class * args.num_classes
        
        model = protoCloud(**model_dict).to(device)
        load_model(args.pretrain_model_pth, model)

        # copy pretrained model label encoder
        if args.cont_train:
            model_encoder = data_info_loader('cell_encoder', os.path.dirname(args.pretrain_model_pth))
            data_info_saver(model_encoder, args.model_dir, 'cell_encoder')
    
    print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    

    # Training
    #######################################################
    args_dict = vars(args)
    if args.model_mode == "train":
        print('\nEnter training')
        if args.model_validation:
            test_y = torch.LongTensor(data.cell_encoder.transform(test_Y))
            result_trend = run_model(model, train_loader,
                                    args.epochs, args.lr, args.optimizer,
                                    args.two_step, args.recon_coef, args.ortho_coef,
                                    test_X = test_X, test_Y = test_y, 
                                    )
        else:
            result_trend = run_model(model, train_loader,
                        args.epochs, args.lr, args.optimizer,
                        args.two_step, args.recon_coef, args.ortho_coef,
                        validate_model = False,
                        )

        # save model to model_dir
        save_model(model, args.model_dir, args.exp_code)
        save_model_dict(model_dict, args.model_dir)
        save_file(result_trend, args.results_dir, args.exp_code, '_trend.npy')  # save loss & accuracy through epochs
        save_file(get_prototypes(model), args.results_dir, args.exp_code, '_prototypes.npy') # save prototypes
        # save model's celltype & gene names
        data_info_saver(data.gene_names, args.model_dir, 'gene_names')
        
        print('\tTraining information saved')
    
    else:
        args.plot_trend = 0
        


    # Predictions
    #######################################################
    args_dict = vars(args)
    if args.model_mode != 'plot':
        predicted = get_predictions(model, test_X, False if args.model_mode == 'apply' else True)

        if args.pretrain_model_pth is not None:
            model_encoder = data_info_loader('cell_encoder', os.path.dirname(args.pretrain_model_pth))
        else:
            model_encoder = data_info_loader('cell_encoder', args.model_dir)
        predicted['pred1'] = model_encoder.inverse_transform(predicted['idx1'])
        predicted['pred2'] = model_encoder.inverse_transform(predicted['idx2'])
        predicted['sim_class'] = model_encoder.inverse_transform(predicted['sim_class'])
        
        if args.new_label: # used pred label, but save orig label as actual label
            test_idx = load_file('_idx.csv', **args)['test_idx'].dropna().astype(int)
            predicted['label'] = data.adata.obs["celltype"][test_idx]
        elif test_Y is None:
            predicted['label'] = []
        else:
            predicted['label'] = test_Y
        predicted = pd.DataFrame(predicted)
        predicted['idx'] = test_idx

        # Prediction comment (certain/ambiguous)
        predicted = identify_TypeError(predicted, data.cell_encoder.classes_)
        
        save_file(predicted, args.results_dir, args.exp_code, '_pred.csv')
        print("\nPredictions saved")


        if args.save_file:
            if model.obs_dist == 'nb' and test_Y is not None:
                try:
                    # log-likelihood
                    ll = model.get_log_likelihood(torch.tensor(test_X[:1000]).to(device), 
                                                torch.tensor(predicted['idx1'][:1000]).to(device)).detach().cpu().numpy()
                    save_file(ll, args.results_dir, args.exp_code, '_ll.npy')
                    print("Avg log-likelihood: ", np.mean(ll))
                    plot_cell_likelihoods(args, ll)
                except Exception as e:
                    print(e)


            latent = get_latent(model, test_X)
            save_file(latent, args.results_dir, args.exp_code, '_latent.npy')
            proto = get_prototypes(model)
            save_file(proto, args.results_dir, args.exp_code, '_prototypes.npy')



    # Visualization
    #######################################################
    print('Visualizing results')
    args_dict = vars(args)
    plot_path = args.plot_dir + args.exp_code

    predicted = load_file(args.results_dir, args.exp_code, '_pred.csv')
    orig = predicted['label']
    pred = predicted['pred1']
    same_label = all(x in np.unique(orig) for x in np.unique(pred))

    
    if args.plot_trend:
        plot_epoch_trend(**args_dict)
    if args.protocorr:
        plot_protocorr_heatmap(args, data)
    
    if args.cm and test_Y is not None:
        plot_confusion_matrix(args)
        plot_prediction_summary(predicted, data.cell_encoder.classes_, 
                                path = plot_path,
                                plot_mis_pred = same_label,
                                **args_dict)
    
    if args.umap or args.two_latent:
        latent_embedding = load_file(args.results_dir, args.exp_code, '_latent.npy')
        proto_embedding = load_file(args.results_dir, args.exp_code, '_prototypes.npy')
    if args.umap:
        plot_latent_embedding(latent_embedding[:, :args.latent_dim//2], 
                            proto_embedding[:, :args.latent_dim//2], 
                            pred = pred, orig = orig, 
                            proto_classes = data.cell_encoder.classes_,
                            path = plot_path,
                            **args_dict)
        # plot_umap_embedding(args, data)
    if args.two_latent:
        plot_latent_embedding(latent_embedding, 
                            proto_embedding, 
                            pred = pred, orig = orig, 
                            proto_classes = data.cell_encoder.classes_,
                            path = plot_path,
                            plot_dim = 2,
                            **args_dict)
    
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
            try:
                generate_PRP_explanations(model_wrapped, 
                                    model.prototype_vectors, 
                                    train_X[:8000], train_Y[:8000], 
                                    data,
                                    model.epsilon, 
                                    **args_dict,
                                    )
            except Exception as e:
                print(f"Error in generate PRP: {e}")
        if args.lrp:
            try:
                generate_LRP_explanations(model_wrapped, 
                                    test_X, test_Y,
                                    data.cell_encoder.classes_, data.gene_names,
                                    model.epsilon, 
                                    **args_dict,
                                    )
            except Exception as e:
                print(f"Error in generate LRP: {e}")

            # #TODO: misclassified genes LRP
            # wrong_pred_idx = np.where(orig != pred)[0]
            # try:
            #     generate_LRP_explanations(model_wrapped, 
            #                         test_X[wrong_pred_idx], test_Y[wrong_pred_idx],
            #                         model.epsilon, 
            #                         args,
            #                         )
            # except Exception as e:
            #     print(f"Error in generate LRP: {e}")
    
    if args.plot_prp:
        print("Ploting PRP visulization")
        plot_prp_dist(data.cell_encoder.classes_, data.gene_names, **args_dict)
        plot_marker_venn_diagram(data.adata, **args_dict)
        plot_top_gene_PRP_dotplot(data.cell_encoder.classes_, data.gene_names, 
                                num_protos = 1, top_num_genes = 10, 
                                celltype_specific = True, save_markers = False, **args_dict)
    if args.plot_lrp:
        print("Ploting LRP visulization")
        plot_lrp_dist(data.cell_encoder.classes_, data.gene_names, **args_dict)


    

    print("Finished!")
        





import argparse

parser = argparse.ArgumentParser(description = 'ProtoCloud')
#######################################################
### ----Path Parameters----
parser.add_argument('--data_dir',       type = str, default = './data/')
parser.add_argument('--model_dir',      type = str, default = './saved_models/')
parser.add_argument('--results_dir',    type = str, default = './results/')
### ----Data Parameters----
parser.add_argument('--dataset_name', type = str, default = 'PBMC_10K')
parser.add_argument('--raw',        type = int, default = 1, choices = [0, 1], help = 'use raw data or normalized data')
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--topngene',   type = int, help = 'number of genes to select')
parser.add_argument('--test_ratio', type = float, default = 0.1)
parser.add_argument('--data_balance', type = int, default = 1, choices = [0, 1], help = 'use weighted sampling for training data or not')
parser.add_argument('--new_label',  type = int, default = 0, choices = [0, 1], help = 'use previous predicted label as target or not')
parser.add_argument('--split',      type = str, default = None, help = "cluster name of test data. if None, use test_ratio")
parser.add_argument('--index_file',  type = str, default = None, help = 'Full path of split indices with columns train_idx and test_idx to load')
#######################################################
### ----Model Parameters----
parser.add_argument('--model_name',       type = str, default = 'protoCloud')
parser.add_argument('--model_mode',  type = str, default = "test", choices = ["train", "test", "apply", "plot"],
                    help = "train model; test: z_mu for pred; apply: all data with reparametrization; plot: load and plot result files using test data")
parser.add_argument('--cont_train',  type = int, default = 0, help = 'Load existing model and continue training')
parser.add_argument('--model_validation',  type = int, default = 1, choices = [0, 1], help = 'validation accuracy in training stage')
parser.add_argument('--pretrain_model_pth',  type = str, help = 'Full path of pre-trained model to load')
parser.add_argument("--encoder_layer_sizes", nargs='+', type=int, help="List of values")
parser.add_argument("--decoder_layer_sizes", nargs='+', type=int, help="List of values")
parser.add_argument('--prototypes_per_class',   type = int, default = 6)
parser.add_argument('--latent_dim',             type = int, default = 20)
# Loss Function Parameters
parser.add_argument('--obs_dist',       type = str, default = 'nb', choices = ['nb', 'normal'])
parser.add_argument('--nb_dispersion',  type = str, default = 'celltype_target', choices = ['celltype_target', 'celltype_pred', 'gene'])
# Optimizer Parameters + Survival Loss Function
parser.add_argument('--two_step',    type = int, default = 1, choices = [0, 1], help = 'use two-step training or not')
parser.add_argument('--early_stopping', type = int, default = 0, choices = [0, 1], help = 'use early stopping training or not')
parser.add_argument('--recon_coef',  type = float, default = 1, help = 'reconstruction loss coefficient')
parser.add_argument('--ortho_coef',  type = float, default = 0.3, help = 'orthogonality loss coefficient')
parser.add_argument('--activation',  type = str, default = 'relu', choices = ['relu'])
parser.add_argument('--use_bias',      type = int, default = 0, choices = [0, 1])
parser.add_argument('--use_dropout',      type = float, default = 0.0)
parser.add_argument('--use_bn',      type = int, default = 1, choices = [0, 1], help = 'use batch normalization or not')
parser.add_argument('--optimizer',   type = str, default = 'AdamW', choices = ['Adam', 'AdamW'])
parser.add_argument('--lr',          type = float, default = 1e-3)
parser.add_argument('--epochs',      type = int, default = 150)
parser.add_argument('--target_accu', type = float, default = 0.7, help = 'mininum target accuracy to save model')
parser.add_argument('--seed',        type = int, default = 7)
#######################################################
### Result plotting Parameters
parser.add_argument('--save_file',  type = int, default = 1, choices = [0, 1], help = 'Save all files')
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
args.exp_code = get_custom_exp_code(**vars(args))
print("Experiment Name:", args.exp_code)


# data directory
makedir(args.data_dir)

# save model directory
model_dir = args.model_dir + args.dataset_name + '/'
args.model_dir = model_dir
makedir(args.model_dir)

# results directory
results_dir = args.results_dir + args.dataset_name + '/'
args.results_dir = results_dir
makedir(results_dir)
args.plot_dir = args.results_dir + 'plots/'
makedir(args.plot_dir )



if args.cont_train:
    args.two_step = 0

if args.visual_result:
    args.plot_trend = 1
    args.cm = 1
    args.umap = 1
    args.two_latent = 1
    args.protocorr = 1
    args.distance_dist = 1

if args.model_mode in ["apply"]:
    args.test_ratio = 1
    args.plot_trend = 0
    args.protocorr = 0
    args.prp = 0
    args.lrp = 0 if args.pretrain_model_pth is not None else args.lrp
        

args.lrp_path = args.results_dir + 'lrp/'
args.prp_path = args.results_dir + 'prp/'
if args.prp:
    makedir(args.prp_path)
    args.plot_prp = 1

if args.lrp:
    makedir(args.lrp_path)
    args.plot_lrp = 1





print('------args---------')
print(args)
with open(args.results_dir + 'args.pkl', 'wb') as f:
    pickle.dump(args, f)
print("\nArgs dict saved")


if __name__ == '__main__':
    main(args)
    print("end script\n\n\n\n\n\n")
    exit()
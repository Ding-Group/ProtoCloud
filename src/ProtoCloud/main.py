import argparse
import os, pickle
import numpy as np
import torch
import pandas as pd
import ProtoCloud

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = 'ProtoCloud')
    #######################################################
    ### ----Path Parameters----
    parser.add_argument('--data_dir',       type = str, default = './data/')
    parser.add_argument('--model_dir',      type = str, default = './saved_models/')
    parser.add_argument('--results_dir',    type = str, default = './results/')
    ### ----Data Parameters----
    parser.add_argument('--dataset_name', type = str, help= 'load <data_dir>/<dataset_name>.h5ad')
    parser.add_argument('--raw',        type = int, default = 1, choices = [0, 1], help = 'use raw data or normalized data')
    parser.add_argument('--preprocess_data',     type = int, default = 0, choices = [0, 1], help = 'preprocess data or not')
    parser.add_argument('--filter_gene_by_counts', type = int, default = 500, help = 'minimum counts for genes')
    parser.add_argument('--filter_cell_by_counts', type = int, default = 1000, help = 'minimum counts for cells')
    parser.add_argument('--normalize_total', type = float, default = 1e4, help = 'total counts for normalization')
    parser.add_argument('--log1p',     type = int, default = 1, choices = [0, 1], help = 'log1p normalization or not')
    parser.add_argument('--topngene',   type = int, help = 'number of genes to select')
    parser.add_argument('--batch_size', type = int, default = 128)
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
    parser.add_argument('--recon_coef',  type = float, default = 10, help = 'reconstruction loss coefficient')
    parser.add_argument('--kl_coef',  type = float, default = 2, help = 'KL divergence loss coefficient')
    parser.add_argument('--ortho_coef',  type = float, default = 1, help = 'orthogonality loss coefficient')
    parser.add_argument('--atomic_coef',  type = float, default = 1, help = 'atomic loss coefficient')
    parser.add_argument('--activation',  type = str, default = 'relu', choices = ['relu'])
    parser.add_argument('--use_bias',      type = int, default = 0, choices = [0, 1])
    parser.add_argument('--use_dropout',      type = float, default = 0.0)
    parser.add_argument('--use_bn',      type = int, default = 1, choices = [0, 1], help = 'use batch normalization or not')
    parser.add_argument('--optimizer',   type = str, default = 'AdamW', choices = ['Adam', 'AdamW'])
    parser.add_argument('--lr',          type = float, default = 1e-3)
    parser.add_argument('--epochs',      type = int, default = 100)
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

    # parser.add_argument('--prp',            type = int, default = 1, choices = [0, 1], help = 'generate all PRP based explanations')
    # parser.add_argument('--plot_prp',       type = int, default = 0, choices = [0, 1], help = 'plot all PRP explanation plots')
    parser.add_argument('--lrp',            type = int, default = 1, choices = [0, 1], help = 'generate LRP based explanations')
    parser.add_argument('--plot_lrp',       type = int, default = 0, choices = [0, 1], help = 'plot LRP explanation plots')


    args = parser.parse_args()


    # set device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4 if torch.cuda.is_available() else 0

    # set random seed
    ProtoCloud.model.seed_torch(torch.device(args.device), args.seed)

    ### Creates Experiment Code from argparse + Folder Name to Save Results
    if args.pretrain_model_pth is not None and not args.cont_train:
        model_exp_code = os.path.basename(args.pretrain_model_pth)[:-4]
        args.exp_code = "_".join(model_exp_code.split("_")[:4] + [args.dataset_name])
    else:
        args.exp_code = ProtoCloud.utils.get_custom_exp_code(**vars(args))
    print("Experiment Name:", args.exp_code)

    # data directory
    ProtoCloud.utils.makedir(args.data_dir)
    # save model directory
    model_dir = os.path.join(args.model_dir, args.dataset_name) + '/'
    args.model_dir = model_dir
    ProtoCloud.utils.makedir(args.model_dir)

    # results directory
    args.results_dir = os.path.join(args.results_dir, args.dataset_name) + '/'
    ProtoCloud.utils.makedir(args.results_dir)
    args.plot_dir = args.results_dir + 'plots/'
    ProtoCloud.utils.makedir(args.plot_dir)
    args.anno_dir = args.results_dir + 'mis_anno/'
    ProtoCloud.utils.makedir(args.anno_dir)


    if args.cont_train:
        args.two_step = 0

    if args.test_ratio == 0:
        args.visual_result = 0
        args.lrp = 0

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
            

    # args.prp_path = args.results_dir + 'prp/'
    # if args.prp:
    #     ProtoCloud.utils.makedir(args.prp_path)
    #     args.plot_prp = 1

    args.lrp_path = args.results_dir + 'lrp/'
    if args.lrp:
        ProtoCloud.utils.makedir(args.lrp_path)
        args.plot_lrp = 1


    print('------args---------')
    print(args)
    with open(args.results_dir + 'args.pkl', 'wb') as f:
        pickle.dump(args, f)
    print("\nArgs dict saved")
    return args




def main() -> None:
    args = parse_args()
    args_dict = vars(args)
    
    ### load dataset
    data = ProtoCloud.data.scRNAData(**args_dict)

    if args.pretrain_model_pth is not None:
        print("Reformat data according to pretrained model")
        # resize data to model input dim
        data.gene_subset(args.pretrain_model_pth)
        data.adata.write(args.results_dir + args.dataset_name + ".h5ad", compression='gzip')
        ProtoCloud.utils.data_info_saver(data.gene_names, args.model_dir, 'gene_names')
    

    print('\nData: ', args.dataset_name)
    train_idx, test_idx = data.get_split_idx(**args_dict)
    print('Training set size: ', len(train_idx))
    print('Test set size: ', len(test_idx))

    (train_X, test_X, train_Y, test_Y) = data.split_data(train_idx, test_idx, **args_dict)
    
    args.batch_size = 1024 if len(train_idx) > 1e5 else 128

    ProtoCloud.data.info_saver(data.cell_encoder, args.model_dir, 'cell_encoder')
    ProtoCloud.data.info_saver(data.gene_names, args.model_dir, 'gene_names')

    if len(test_idx) != 0:
        test_X = torch.Tensor(test_X)
        
    args.model_validation = 0 if (len(test_idx) == 0 or test_Y is None or args.new_label) else args.model_validation
    model_encoder = data.cell_encoder


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
        model = ProtoCloud.protoCloud(**model_dict).to(args.device)
        # print(model)
        if args.cont_train or args.model_mode != "train":
            print('\nLoading existing model')
            ProtoCloud.model.load_model(args.model_dir + args.exp_code + '.pth', model)
    
    else:   # args.pretrain_model_pth not None
        print('\nLoad existing model from: \n\t', args.pretrain_model_pth)
        model_dict = ProtoCloud.model.load_model_dict(os.path.dirname(args.pretrain_model_pth), args.device)
        print("Loaded model parameters: \n\t", model_dict)

        args.input_dim = model_dict["input_dim"]
        args.num_classes = model_dict["num_classes"]
        args.prototypes_per_class = model_dict["num_prototypes_per_class"]
        args.num_prototypes = args.prototypes_per_class * args.num_classes
        
        model = ProtoCloud.protoCloud(**model_dict).to(args.device)
        ProtoCloud.model.load_model(args.pretrain_model_pth, model)

        # copy pretrained model label encoder
        if args.cont_train:
            model_encoder = ProtoCloud.data.info_loader('cell_encoder', os.path.dirname(args.pretrain_model_pth))
            ProtoCloud.data.info_saver(model_encoder, args.model_dir, 'cell_encoder')
    
    print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    

    # Training
    #######################################################
    args_dict = vars(args)
    if args.model_mode == "train":
        print('\nEnter training')
        if args.model_validation:
            test_y = torch.LongTensor(data.cell_encoder.transform(test_Y))
            result_trend = ProtoCloud.model.run_model(model, 
                                    train_X, train_Y,
                                    test_X = test_X, test_Y = test_y, 
                                    **args_dict
                                    )
        
        else:
            result_trend = ProtoCloud.model.run_model(model, 
                                    train_X, train_Y,
                                    validate_model = False,
                                    **args_dict,
                                    )

        # save model to model_dir
        ProtoCloud.model.save_model(model, args.model_dir, args.exp_code)
        ProtoCloud.model.save_model_dict(model_dict, args.model_dir)
        ProtoCloud.utils.save_file(result_trend, args.results_dir, args.exp_code, '_trend.npy')
        ProtoCloud.utils.save_file(ProtoCloud.model.get_prototypes(model), args.results_dir, args.exp_code, '_prototypes.npy')



        # get training data class threshold
        predicted = ProtoCloud.model.get_predictions(model, train_X)
        if model.obs_dist == 'nb':
            ll = ProtoCloud.model.get_log_likelihood(model, train_X, 
                                                train_Y if model.nb_dispersion == "celltype_target" else None)
            predicted['ll'] = ll
        predicted = ProtoCloud.utils.process_prediction_file(predicted, model_encoder,
                                            model_encoder.inverse_transform(train_Y))
        cls_threshold = ProtoCloud.utils.get_cls_threshold(predicted)
        ProtoCloud.data.info_saver(cls_threshold, args.model_dir, 'cls_threshold')
        print('\tTraining information saved')
    
    else:
        args.plot_trend = 0
        


    # Predictions
    #######################################################
    args_dict = vars(args)
    if args.model_mode != 'plot' and args.test_ratio != 0:
        # predicted = get_predictions(model, test_X, False if args.model_mode == 'apply' else True)
        predicted = ProtoCloud.model.get_predictions(model, test_X, False)
        if model.obs_dist == 'nb' and test_Y is not None:
            try:
                predicted['ll'] = ProtoCloud.model.get_log_likelihood(model, test_X, 
                                        Y = predicted['idx1'] if args.nb_dispersion == "celltype_target" else None
                                        )
                print("Avg log-likelihood: ", np.mean(predicted['ll']))
                ProtoCloud.vis.plot_cell_likelihoods(args, predicted['ll'])
            except Exception as e:
                print(e)

        if args.pretrain_model_pth is not None:
            model_encoder = ProtoCloud.data.info_loader('cell_encoder', os.path.dirname(args.pretrain_model_pth))
            cls_threshold = ProtoCloud.data.info_loader('cls_threshold', os.path.dirname(args.pretrain_model_pth))
        else:
            model_encoder = ProtoCloud.data.info_loader('cell_encoder', args.model_dir)
        
        if args.new_label: # used pred label, but save orig label as actual label
            indices = ProtoCloud.utils.load_file(file_ending='_idx.csv', **args_dict)
            test_idx = indices['test_idx'].dropna().values.astype(int) 
            label = data.adata.obs["celltype"][test_idx].values if ("celltype" in data.adata.obs.columns) else None
            print(label)
        elif test_Y is None:
            label = None
        else:
            label = test_Y
        
        predicted['idx'] = test_idx
        predicted = pd.DataFrame(predicted)
        ProtoCloud.utils.save_file(predicted, args.results_dir, args.exp_code, '_pred0.csv')
        # transform labels, add prediction comment (certain/ambiguous)
        predicted = ProtoCloud.utils.process_prediction_file(predicted, model_encoder, label, 
                                            model_dir = os.path.dirname(args.pretrain_model_pth) if args.pretrain_model_pth is not None else args.model_dir)
        
        ProtoCloud.utils.save_file(predicted, args.results_dir, args.exp_code, '_pred.csv')
        print("\nPredictions saved")


        if args.save_file:
            latent = ProtoCloud.model.get_latent(model, test_X)
            ProtoCloud.utils.save_file(latent, args.results_dir, args.exp_code, '_latent.npy')
            proto = ProtoCloud.model.get_prototypes(model)
            ProtoCloud.utils.save_file(proto, args.results_dir, args.exp_code, '_prototypes.npy')



    # Visualization
    #######################################################
    print('Visualizing results')
    args_dict = vars(args)
    plot_path = args.plot_dir + args.exp_code

    predicted = ProtoCloud.utils.load_file(args.results_dir, args.exp_code, '_pred.csv')
    orig = predicted['label']
    pred = predicted['pred1']
    same_label = all(x in np.unique(orig) for x in np.unique(pred))

    if args.plot_trend:
        ProtoCloud.vis.plot_epoch_trend(**args_dict)
    if args.protocorr:
        ProtoCloud.vis.plot_protocorr_heatmap(args, data)
    
    if args.cm and test_Y is not None:
        ProtoCloud.vis.plot_confusion_matrix(args)
    
        ProtoCloud.vis.plot_prediction_summary(predicted, data.cell_encoder.classes_, 
                                path = plot_path,
                                plot_mis_pred = same_label,
                                **args_dict)
    
    if args.umap or args.two_latent:
        latent_embedding = ProtoCloud.utils.load_file(args.results_dir, args.exp_code, '_latent.npy')
        # latent_embedding = get_latent(model, data.to_dense(data.adata, raw=True))
        proto_embedding = ProtoCloud.utils.load_file(args.results_dir, args.exp_code, '_prototypes.npy')
    if args.umap:
        ProtoCloud.vis.plot_latent_embedding(latent_embedding[:, :args.latent_dim//2], 
                            proto_embedding[:, :args.latent_dim//2], 
                            pred = pred, orig = orig, 
                            proto_classes = data.cell_encoder.classes_,
                            path = plot_path,
                            **args_dict)
        # plot_umap_embedding(args, data)
    if args.two_latent:
        ProtoCloud.vis.plot_latent_embedding(latent_embedding, 
                            proto_embedding, 
                            pred = pred, orig = orig, 
                            proto_classes = data.cell_encoder.classes_,
                            path = plot_path,
                            plot_dim = 2,
                            **args_dict)
    
    if args.distance_dist:
        ProtoCloud.vis.plot_distance_to_prototypes(args, data)


    #######################################################
    # LRP based explanations & marker gene selection
    if args.lrp:
        print("\nGenerating model explanations")
        wrapper = ProtoCloud.lrp.model_canonized()
        # construct the model for generating LRP based explanations
        model_wrapped = ProtoCloud.protoCloud(**model_dict).to(args.device)
        # replace layers in model_wrapped with customized LRP layers
        wrapper.copyfrommodel(model_wrapped, 
                              model)
        print("\tModel wrapped")

        if args.lrp:
            ProtoCloud.lrp.generate_LRP_explanations(model_wrapped, 
                                test_X, test_Y,
                                data.cell_encoder.classes_, data.gene_names,
                                model.epsilon, 
                                **args_dict,
                                )


    # if args.plot_prp:
    #     print("Ploting PRP visulization")
    #     ProtoCloud.vis.plot_prp_dist(data.cell_encoder.classes_, data.gene_names, **args_dict)
    #     ProtoCloud.vis.plot_marker_venn_diagram(data.adata, **args_dict)
    #     ProtoCloud.vis.plot_top_gene_PRP_dotplot(data.cell_encoder.classes_, data.gene_names, 
    #                             num_protos = 1, top_num_genes = 10, 
    #                             celltype_specific = False, save_markers = False, **args_dict)
    #     ProtoCloud.vis.plot_top_gene_PRP_dotplot(data.cell_encoder.classes_, data.gene_names, 
    #                             num_protos = 1, top_num_genes = 10, 
    #                             celltype_specific = True, save_markers = False, **args_dict)
    if args.plot_lrp:
        print("Ploting LRP visulization")
        ProtoCloud.vis.plot_lrp_dist(data.cell_encoder.classes_, data.gene_names, **args_dict)


    
    print("Finished!")


if __name__ == "__main__":
    main()
# from torchvision import datasets
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import show
from sklearn.preprocessing import MinMaxScaler

from .lrp_general import *
from ..utils import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_workers = 4 if torch.cuda.is_available() else 0

import ProtoCloud.glo as glo
EPS = glo.get_value('EPS')
LRP_FILTER_TOP_K = glo.get_value('LRP_FILTER_TOP_K')


class  Modulenotfounderror(Exception):
  pass


### LRP parameters
lrp_params_def1 = {
    'linear_eps': EPS,
    'linear_gamma': 0.1,
    'linear_ignorebias': True,
    }

lrp_layer2method = {
    'nn.ReLU':          relu_wrapper_fct,
    'nn.BatchNorm1d':   batchnorm1d_wrapper_fct,
    # 'nn.Linear':        linearlayer_eps_wrapper_fct,
    # 'nn.Linear':        linearlayer_gamma_wrapper_fct,
    'nn.Linear':        linearlayer_Alpha1Beta0_wrapper_fct, # equals to LRP-Z+ rule
    # 'nn.Linear':        linearlayer_Alpha2Beta1_wrapper_fct,
    # 'nn.Linear':        linearlayer_absZ_wrapper_fct,
    }


class model_canonized():
    def __init__(self):
        super(model_canonized, self).__init__()
    def setbyname(self, model, name, value):
        """
        Update the layers in the new model to customized layers
        Args:
            model: new protoCloud model
            name: name of the target layer to be replaced
            value: warpped (customized layer)
        """
        def _iteratset(obj, components, value):
            if not hasattr(obj, components[0]):
                return False
            elif len(components) == 1:
                setattr(obj, components[0], value)  # setattr(x, 'y', v) is equivalent to x.y = v''
                # print('found!!', components[0])
                # exit()
                return True
            else:
                nextobj = getattr(obj, components[0])   # getattr(x, 'y') is equivalent to x.y
                return _iteratset(nextobj, components[1:], value)

        components = name.split('.')
        success = _iteratset(model, components, value)
        # print("after setbyname:",success)
        return success


    ############# Main Function ###############
    def copyfrommodel(self, model, net, lrp_layer2method=lrp_layer2method, lrp_params=lrp_params_def1, single_linear=False):
        """
        Args:
            model: new protoCloud.features
            net: trained protoCloud.features
            lrp_params: {'linear_eps': 1e-6,}
            lrp_layer2method: lrp layer to method mapping (see lrp_general.py)
            single_linear: whether wait for update along with BN
        """
        # Model: {(linear-bn-relu) * n } - linear
        # means: when encounter bn, find the linear before -- implementation dependent

        updated_layers_names = []

        last_src_module_name = None
        last_src_module = None

        # replace the layers in the new model with customized passing
        for src_module_name, src_module in net.named_modules():
            foundsth = False

            if isinstance(src_module, nn.Linear):
                # copy linear layers
                foundsth = True
                if (not ("encoder" in src_module_name or "decoder" in src_module_name)):
                    # print("\tsingle_linear:", src_module_name)
                    wrapped = get_lrpwrapperformodule(copy.deepcopy(src_module), 
                                                      lrp_params, 
                                                      lrp_layer2method)
                    if False == self.setbyname(model, src_module_name, wrapped):
                        raise Modulenotfounderror("could not find " + src_module_name + " in target model")
                    updated_layers_names.append(src_module_name)
                else:
                    last_src_module_name = src_module_name
                    last_src_module = src_module
            # end of if

            elif isinstance(src_module, nn.BatchNorm1d):
                # linear-bn chain: when encounter bn, find the linear before
                foundsth = True

                last_linear = copy.deepcopy(last_src_module)
                # Step 1: Fuse BN into linear
                last_linear = bnafterlinear_overwrite_intolinear(last_linear, bn=src_module)
                # Step 2: Overwrite/wrap the linear
                wrapped = get_lrpwrapperformodule(last_linear, 
                                                  lrp_params, 
                                                  lrp_layer2method,
                            )
                if False == self.setbyname(model, last_src_module_name, wrapped):
                    raise Modulenotfounderror("could not find " + last_src_module_name + " in target model")
                updated_layers_names.append(last_src_module_name)

                # Step 3: reset BN to identity
                wrapped = get_lrpwrapperformodule(resetbn(src_module), # reset bn to default first
                                                  lrp_params, 
                                                  lrp_layer2method)
                if False == self.setbyname(model,src_module_name, wrapped):
                    raise Modulenotfounderror("could not find " + src_module_name + " in target model")
                updated_layers_names.append(src_module_name)


            elif isinstance(src_module, nn.ReLU):
                wrapped = get_lrpwrapperformodule(src_module, lrp_params, lrp_layer2method)

                if False == self.setbyname(model,src_module_name, wrapped):
                    raise Modulenotfounderror("could not find " + src_module_name + " in target model")
                updated_layers_names.append(src_module_name)
            
            else:
                # module
                module_name = src_module_name
                num_layers = len(list(src_module.children()))


        # for target_module_name, target_module in net.named_modules():
        #     if target_module_name not in updated_layers_names:
        #         print('\tNot updated modules:', target_module_name)



def x_prp(model, input, prototype, epsilon):
    cell = 0
    x = torch.Tensor(input).to(device)
    x.requires_grad = True

    with torch.enable_grad():
        if model.raw_input:
            x = torch.log(x + 1)
            x.retain_grad()
        zx_mean = model.encoder(x)
        zx_mean = model.z_mean(zx_mean)

        half_dim = prototype.shape[0] // 2
        d = (zx_mean[:, :half_dim] - prototype[:half_dim])**2
        R_zx = 1 / (d + epsilon)            # relavance of prototype to z_mu

        R_zx.backward(torch.ones_like(R_zx))            # backward through encoder
        rel = x.grad.data.to('cpu').detach().numpy()    # gradient of input
        # print("rel.shape", rel.shape)   # torch.Size([n, 3346])
        # print("\tPRP:", np.max(rel), np.min(rel))

    rel_mean = np.mean(rel, axis = 0)   # [1, 3346]
    del x, zx_mean, d, R_zx, rel
    torch.cuda.empty_cache()
    return rel_mean

def x_lrp(model, input):
    cell = 0
    x = torch.Tensor(input).to(device)
    x.requires_grad = True

    with torch.enable_grad():
        if model.raw_input:
            x = torch.log(x + 1)
            x.retain_grad()
        zx_mean = model.encoder(x)
        zx_mean = model.z_mean(zx_mean)
        s_px = model.calc_sim_scores(zx_mean)
        R_yx = model.classifier(s_px)

        R_yx.backward(torch.ones_like(R_yx))    # backward through encoder
        rel = x.grad.data.to('cpu').detach().numpy()  # gradient of input [n, 3346]
        # print("\tLRP:", np.max(rel), np.min(rel))
    
    del x, zx_mean, s_px, R_yx
    torch.cuda.empty_cache()
    return rel

def z_recon(model, input):
    cell = 0
    x = torch.Tensor(input).to(device)
    x.requires_grad = True

    with torch.enable_grad():
        if model.raw_input:
            x = torch.log(x + 1)
            x.retain_grad()
        z = model.encoder(x)
        zx_mean = model.z_mean(z)
        zx_mean = torch.nn.Parameter(zx_mean, requires_grad=True) 
        px = model.decoder(zx_mean)
        px_mu = model.px_mean(px)

        px_mu.backward(torch.ones_like(px_mu))    # backward through encoder
        rel = zx_mean.grad.data.to('cpu').detach().numpy()  # gradient of input

    rel_sum = np.sum(rel, axis = 0)   # [1, latent]
    return rel_sum



## Generating protoypical explanations for each prototypes/input data
def generate_PRP_explanations(model,    # model_wrapped
                          prototypes,   # protoCloud.prototype_vectors
                          train_X, train_Y, 
                          data,
                          epsilon, 
                          num_classes, prototypes_per_class,
                          prp_path=None, exp_code=None, **kwargs):
    model.eval()

    gene_names = data.gene_names
    cell_types = data.cell_encoder.classes_
    file_name = "_"+exp_code+"_relgenes.npy"
    # cell_types = [str(i) for i in range(num_classes)]
    print("Generating PRP explainations:")

    proto_count = 0
    scaler = MinMaxScaler()
    ava_type = []
    for c in range(num_classes):
        try:
            idx = np.where(train_Y == c)[0]
            class_prp = []
            # for each class, get the PRP genes of each prototype
            for pno in range(prototypes_per_class):
                rel_mean = x_prp(model, train_X[idx, :], 
                                prototypes[proto_count+pno, :], epsilon)
                class_prp.append(rel_mean)
            proto_count += prototypes_per_class
            class_prp = np.stack(class_prp, axis = 0)     # (n_prototypes, n_genes)
            class_prp = scaler.fit_transform(class_prp.T).T
            # save celltype corresponding PRP genes
            path = prp_path + cell_types[c].replace("/", " OR ") + file_name
            save_file(class_prp, save_path = path)
            ava_type.append(cell_types[c])
        except Exception as e:
            print(c, len(np.where(train_Y == c)[0]))
            print(f"Error in generate PRP for {cell_types[c]}: {e}")
    
    df = rank_HRG(ava_type, gene_names, prp_path, filename = file_name)
    save_path = prp_path + "PRP_rank.csv"
    save_file(df, save_path=save_path)
    print("Saved PRP genes for each class")



def generate_LRP_explanations(model,    # model_wrapped
                          test_X, test_Y, cell_types, gene_names,
                          epsilon, lrp_path=None, exp_code=None, **kwargs):
    model.eval()

    # cell_types = np.unique(test_Y)
    print("Generating LRP explainations:")
    scaler = MinMaxScaler()
    for c in cell_types:
        try:
            idx = np.where(test_Y == c)[0]
            # save celltype corresponding LRP genes
            rel = x_lrp(model, test_X[idx, :])   # [n, G]
            path = lrp_path + c.replace("/", " OR ") + '_'
            save_file(rel, path, exp_code, "_lrp")
            rel_sum = np.sum(rel, axis = 0).reshape(1, -1)   # (1, G)
            rel_sum = scaler.fit_transform(rel_sum.T).T.flatten()
            path = lrp_path + c.replace("/", " OR ") + '_'
            save_file(rel_sum, path, exp_code, "_relgenes")

            # save celltype corresponding recon latents
            rel_sum = z_recon(model, test_X[idx, :])
            save_file(rel_sum, path, exp_code, "_latents")
        except Exception as e:
            print(f"Error in generate LRP for {c}: {e}")
        
    print("Saved LRP genes for each class")
        

def save_prp_genes(rel, gene_names, write_path):
    """
    rel: rel.to('cpu')   # gradient of input
    gene_names: data.gene_names
    write_path: prp_result_path + str(pno) + '_class' + str(c) + '/'
    """
    rel = rel.detach().numpy()  # torch.Size([n, 3346])
    normalized_rel = 2 * ((rel - rel.min()) / (rel.max() - rel.min())) - 1    # [-1,1]

    # get top 100 freq genes for the class c prototype pno
    idx = -(normalized_rel).argsort()
    top = (idx < 100).astype(int)
    freq = np.sum(top, axis = 0)
    
    gene_rel_mean = np.mean(normalized_rel, axis = 0) # (1, 3346)

    return gene_rel_mean

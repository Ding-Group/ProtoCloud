import torch
import torch.nn as nn

import copy
import torch.nn.functional as F

import ProtoCloud.glo as glo
EPS = glo.get_value('EPS')
LRP_FILTER_TOP_K = glo.get_value('LRP_FILTER_TOP_K')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_workers = 4 if torch.cuda.is_available() else 0



#######################################################
#######################################################
# wrappers for autograd type modules
#######################################################
#######################################################
class zeroparam_wrapper_class(nn.Module):
  def __init__(self, module, autogradfunction):
    super(zeroparam_wrapper_class, self).__init__()
    self.module = module
    self.wrapper = autogradfunction

  def forward(self,x):
    y = self.wrapper.apply(x, self.module)
    return y


class oneparam_wrapper_class(nn.Module):
  def __init__(self, module, autogradfunction, parameter1):
    super(oneparam_wrapper_class, self).__init__()
    self.module = module
    self.wrapper = autogradfunction
    self.parameter1 = parameter1

  def forward(self, x):
    y = self.wrapper.apply(x, self.module, self.parameter1)
    return y


class twoparam_wrapper_class(nn.Module):
  def __init__(self, module, autogradfunction, parameter1, parameter2):
    super(twoparam_wrapper_class, self).__init__()
    self.module = module
    self.wrapper = autogradfunction
    self.parameter1 = parameter1
    self.parameter2 = parameter2

  def forward(self, x):
    y = self.wrapper.apply(x, self.module, self.parameter1, self.parameter2)
    return y


class  lrplookupnotfounderror(Exception):
  pass



#######################################################
#######################################################
# def get_lrpwrapperformodule(module, 
#                             lrp_params, 
#                             lrp_layer2method, 
#                             ):

#   autogradfunction = lrp_layer2method
#   if isinstance(module, nn.ReLU):
#     return zeroparam_wrapper_class(module, 
#                                    autogradfunction=autogradfunction)


#   elif isinstance(module, nn.BatchNorm1d):
#     return zeroparam_wrapper_class(module, 
#                                    autogradfunction=autogradfunction)


#   elif isinstance(module, nn.Linear):
#     if type(autogradfunction) == linearlayer_eps_wrapper_fct:
#       return oneparam_wrapper_class(module, 
#                                     autogradfunction = autogradfunction, 
#                                     parameter1 = lrp_params['linear_eps'])
#     elif type(autogradfunction) == linearlayer_gamma_wrapper_fct:
#       return oneparam_wrapper_class(module, 
#                                     autogradfunction = autogradfunction, 
#                                     parameter1 = lrp_params['linear_gamma'] )
#     elif type(autogradfunction) == linearlayer_Alpha1Beta0_wrapper_fct:
#        return oneparam_wrapper_class(module, 
#                                     autogradfunction = autogradfunction, 
#                                     parameter1 = lrp_params['apply_filter'])
#     elif type(autogradfunction) == linearlayer_Alpha2Beta1_wrapper_fct:
#        return oneparam_wrapper_class(module, 
#                                     autogradfunction = autogradfunction, 
#                                     parameter1 = lrp_params['linear_ignorebias'])
#     # elif type(autogradfunction) == linearlayer_absZ_wrapper_fct:
#     #    return oneparam_wrapper_class(module, 
#     #                                 autogradfunction = autogradfunction, 
#     #                                 parameter1 = lrp_params['linear_ignorebias'] )
#     # elif type(autogradfunction) == linearlayer_patternAttribution_wrapper_fct:

#   else:
#     raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", module)

def get_lrpwrapperformodule(module, 
                            lrp_params, lrp_layer2method):
  """
  Args:
    module: a PyTorch module (e.g., nn.Linear, nn.ReLU)
    lrp_params: dict of LRP parameters
    lrp_layer2method: class name of LRP method (autograd function)  
  """
  autogradfunction = lrp_layer2method()
  
  if isinstance(module, nn.ReLU):
    return zeroparam_wrapper_class(module, 
                                   autogradfunction=autogradfunction)

  elif isinstance(module, nn.BatchNorm1d):
    return zeroparam_wrapper_class(module, 
                                   autogradfunction=autogradfunction)

  elif isinstance(module, nn.Linear):
    if type(autogradfunction) == linearlayer_eps_wrapper_fct:
      return oneparam_wrapper_class(module, 
                                    autogradfunction = autogradfunction, 
                                    parameter1 = lrp_params['linear_eps'] )
    elif type(autogradfunction) == linearlayer_gamma_wrapper_fct:
      return oneparam_wrapper_class(module, 
                                    autogradfunction = autogradfunction, 
                                    parameter1 = lrp_params['linear_gamma'] )
    elif type(autogradfunction) == linearlayer_Alpha1Beta0_wrapper_fct:
       return oneparam_wrapper_class(module, 
                                    autogradfunction = autogradfunction, 
                                    parameter1 = lrp_params['apply_filter'] )
    elif type(autogradfunction) == linearlayer_Alpha2Beta1_wrapper_fct:
       return oneparam_wrapper_class(module, 
                                    autogradfunction = autogradfunction, 
                                    parameter1 = lrp_params['linear_ignorebias'] )
    elif type(autogradfunction) == linearlayer_abs_wrapper_fct:
       return zeroparam_wrapper_class(module, 
                                    autogradfunction = autogradfunction)
    else:
        raise RuntimeError("Implementation DNE: ", autogradfunction)

  else:
    raise lrplookupnotfounderror("Found no dictionary entry for this module name:", module)


#######################################################
#######################################################
#canonization functions
#######################################################
#######################################################

def resetbn(bn):
  assert (isinstance(bn,nn.BatchNorm1d))

  bnc=copy.deepcopy(bn)
  bnc.reset_parameters()

  return bnc


#vanilla fusion conv-bn --> conv(updatedparams)
def bnafterlinear_overwrite_intolinear(linear, bn):
    # Equation(Page 11): https://github.com/AlexBinder/LRP_Pytorch_Resnets_Densenet/blob/master/canonization_doc.pdf
    #  weight and bias in _BatchNorm are the gamma and beta in the documentation of torch.nn.BatchNorm1d
    assert (isinstance(bn, nn.BatchNorm1d))
    assert (isinstance(linear, nn.Linear))

    s = (bn.running_var + bn.eps) ** 0.5
    w = bn.weight
    bias = bn.bias
    mu = bn.running_mean

    linear.weight = torch.nn.Parameter(linear.weight * (w / s).unsqueeze(1))

    if linear.bias is None:
      linear.bias = torch.nn.Parameter((0 - mu) * (w / s) + bias)
    else:
      linear.bias = torch.nn.Parameter((linear.bias - mu) * (w / s) + bias)

    return linear




#######################################################
#######################################################
# autograd type modules
#######################################################
#######################################################

class relu_wrapper_fct(torch.autograd.Function): 
    # to be used with generic_activation_pool_wrapper_class(module,this)
    @staticmethod
    def forward(ctx, x, module):
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
    

class batchnorm1d_wrapper_fct(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, x, module):
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None




class sim_score_eps_wrapper_fct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, module, eps=1e-9):
        prototypes = module.prototype_vectors.detach().clone()
        scale = module.scale.detach().clone()
        latent_dim_half = module.latent_dim // 2
        
        ctx.save_for_backward(x, prototypes, scale)
        ctx.latent_dim_half = latent_dim_half
        ctx.eps = eps
        
        with torch.no_grad():
             d = torch.cdist(x[:, :latent_dim_half], 
                             prototypes[:, :latent_dim_half], p=2)
             sim_scores = 1.0 / (torch.square(d * scale) + 1.0)
             
        return sim_scores

    @staticmethod
    def backward(ctx, grad_output):
        input_, prototypes, scale = ctx.saved_tensors
        eps = ctx.eps
        latent_dim_half = ctx.latent_dim_half

        X = input_.clone().detach().requires_grad_(True)
        relevance_output = grad_output.clone().detach()
        
        print('Init relevance: ', relevance_output.min().item(), relevance_output.max().item())

        # with torch.enable_grad():
        #     d = torch.cdist(X[:, :latent_dim_half], 
        #                     prototypes[:, :latent_dim_half], p=2)
        #     Z = 1.0 / (torch.square(d * scale) + 1.0)

        # S = relevance_output / Z.clone().detach()
        # Z.backward(S)
        # grad_input = X.grad.data
        # return torch.abs(R), None, None
        
        # mannual compute sim backward
        # grad_input = 2*relevance_output * torch.abs(X[:, :latent_dim_half] - prototypes[:, :latent_dim_half]) / \
        #               (torch.cdist(X[:, :latent_dim_half], prototypes[:, :latent_dim_half], p=2) + 1)**2
        X_half = X[:, :latent_dim_half]
        P_half = prototypes[:, :latent_dim_half]
        diff = X_half.unsqueeze(1) - P_half.unsqueeze(0)

        # derive of sim score w.r.t. input X
        # d/dx = -1 * R_out * scale^2 * 2 / (denom^2)
        dist_sq = torch.sum(diff ** 2, dim=2) 
        denominator = 1.0 + (scale ** 2) * dist_sq
        coeff = -2 * (scale ** 2) * relevance_output / (denominator ** 2)
        # take abs to avoid flipping sign issue
        # grads_per_proto = coeff.unsqueeze(2) * torch.abs(diff)
        grads_per_proto = coeff.unsqueeze(2) * diff
        
        # collect gradients from all prototypes
        # [Batch, Dim]
        grad_input_half = torch.sum(grads_per_proto, dim=1)
        # fill zeros for the rest dimensions
        grad_input = torch.zeros_like(X)
        grad_input[:, :latent_dim_half] = grad_input_half
        
        R = X.data * grad_input

        print('sim_score rel backward: ', R.min().item(), R.max().item())
        # print('sim_score rel backward: ', torch.abs(R).min().item(), torch.abs(R).max().item())
        return R, None, None


#######################################################
#######################################################
#######################################################
### rhos & incrs: https://github.com/fhvilshoj/TorchLRP/issues/1
## rhos
identity_fn  = lambda w, b: (w, b)

def gamma_fn(gamma): 
    def _gamma_fn(w, b):
        w = w + w * torch.max(torch.tensor(0., device = w.device), w) * gamma
        if b is not None: b = b + b * torch.max(torch.tensor(0., device = b.device), b) * gamma
        return w, b
    return _gamma_fn

## incrs
add_epsilon_fn = lambda e: lambda x:   x + ((x > 0).float() * 2 - 1) * e



#######################################################
# lineareps_wrapper_fct
# LRP-epsilon Rule
class linearlayer_eps_wrapper_fct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, module, eps):
        #stash module config params and trainable params
        propertynames, values = _configvalues_totensorlist(module)
        epsTensor= torch.tensor([eps], dtype = torch.float32, device = x.device) 

        weight = module.weight.data.clone()
        if module.bias is None:
          bias = None
        else:
          bias = module.bias.data.clone()

        ctx.save_for_backward(x, weight, bias, epsTensor, *values)
        ctx.rho = identity_fn

        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, bias, epsTensor, *values = ctx.saved_tensors
        rho = ctx.rho
        paramsdict = tensorlist_todict(values)
        eps = epsTensor.item()
        
        # set up module weights and bias
        weight, bias = rho(weight, bias)
        if bias is None:
          module = nn.Linear(**paramsdict, bias = False )
        else:
          module = nn.Linear(**paramsdict, bias = True )
          module.bias = torch.nn.Parameter(bias)
        module.weight = torch.nn.Parameter(weight)

        X = input_.clone().detach().requires_grad_(True)
        
        relevance_output = grad_output.clone().detach()

        # Rule
        with torch.enable_grad():
          Z = module(X)
        S = safe_divide(relevance_output, Z.clone().detach(), eps0 = eps, eps = eps)
        Z.backward(S)
        R = X.data * X.grad.data

        # print('linaer custom R', R.shape )
        #exit()
        return R, None, None



# LRP-gamma Rule
class linearlayer_gamma_wrapper_fct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, module, gamma):
        #stash module config params and trainable params
        propertynames, values = _configvalues_totensorlist(module)

        weight = module.weight.data.clone()
        if module.bias is None:
          bias = None
        else:
          bias = module.bias.data.clone()

        ctx.save_for_backward(x, weight, bias, *values)
        ctx.rho = gamma_fn(gamma)   # gamma rule!

        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, bias, *values = ctx.saved_tensors
        rho = ctx.rho
        paramsdict = tensorlist_todict(values)
        
        # set up module weights and bias
        weight, bias = rho(weight, bias)
        if bias is None:
          module = nn.Linear(**paramsdict, bias = False )
        else:
          module = nn.Linear(**paramsdict, bias = True )
          module.bias = torch.nn.Parameter(bias)
        module.weight = torch.nn.Parameter(weight)

        X = input_.clone().detach().requires_grad_(True)
        
        relevance_output = grad_output.clone().detach()

        # Rule
        with torch.enable_grad():
          Z = module(X)
        S = safe_divide(relevance_output, Z.clone().detach(), eps0 = EPS, eps = 0)
        Z.backward(S)
        R = X.data * X.grad.data

        # print('linaer custom R', R.shape )
        #exit()
        return R, None, None



# LRP-Alpha1Beta0 Rule == Z_plus rule
class linearlayer_Alpha1Beta0_wrapper_fct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, module, apply_filter):
        # #stash module config params and trainable params
        propertynames, values = _configvalues_totensorlist(module)

        weight = module.weight.data.clone()
        if module.bias is None:
          bias = None
        else:
          bias = module.bias.data.clone()

        applyFilterTensor=torch.tensor([apply_filter], dtype = torch.bool, device = module.weight.device)

        ctx.save_for_backward(x, weight, bias, applyFilterTensor, *values)
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        # input_, weight, bias, epstensor, *values = ctx.saved_tensors
        input_, weight, bias, applyFilterTensor, *values = ctx.saved_tensors
        paramsdict = tensorlist_todict(values)
        apply_filter = applyFilterTensor.item()
        
        # set up module weights and bias
        if bias is None:
          module = nn.Linear(**paramsdict, bias = False )
        else:
          module = nn.Linear(**paramsdict, bias = True )
          module.bias = torch.nn.Parameter(bias)
        module.weight = torch.nn.Parameter(weight)

        X = input_.clone().detach().requires_grad_(True)
        relevance_output = grad_output.clone().detach()
        
        if apply_filter:
          relevance_output = relevance_filter(relevance_output, top_k_percent = LRP_FILTER_TOP_K)
        # keeps k% of the largest tensor elements
        # relevance_output = relevance_filter(relevance_output, top_k_percent = LRP_FILTER_TOP_K)

        sel = weight > 0
        zeros = torch.zeros_like(weight)

        weight_pos       = torch.where(sel,  weight, zeros)
        weight_neg       = torch.where(~sel, weight, zeros)

        input_pos         = torch.where(X >  0, X, torch.zeros_like(X))
        input_neg         = torch.where(X <= 0, X, torch.zeros_like(X))

        def _f(X1, X2, W1, W2): 
          Z1  = F.linear(X1, W1, bias=None) 
          Z2  = F.linear(X2, W2, bias=None)
          Z   = Z1 + Z2

          rel_out = relevance_output / (Z + (Z==0).float()* EPS)
          t1 = F.linear(rel_out, W1.t(), bias=None) 
          t2 = F.linear(rel_out, W2.t(), bias=None)
          r1  = t1 * X1
          r2  = t2 * X2

          return r1 + r2
        
        R1 = _f(input_pos, input_neg, weight_pos, weight_neg)
        R2 = _f(input_neg, input_pos, weight_pos, weight_neg)

        # # Rule
        # pnlinear = posneglinear(module)
        # nplinear = negposlinear(module)
        # def _f(X, module, relevance_output):
        #   with torch.enable_grad():
        #     Z = module(X)
        #     print("Z min/max: ", torch.min(Z), torch.max(Z))
        #   # S = safe_divide(relevance_output, Z.clone().detach(), eps0 = EPS, eps = 0)
        #   S = (relevance_output / (Z + EPS * (Z == 0).to(Z))).data
        #   Z.backward(S)
        #   R = X.data * X.grad.data
        #   return R
        # R1 = _f(X, pnlinear, relevance_output)
        # print("Alpha: ", torch.min(R1), torch.max(R1))
        # R2 = _f(X, nplinear, relevance_output)
        # print("Beta: ", torch.min(R2), torch.max(R2))

        R = R1 * 1 - R2 * 0       # alpha=1, beta=0

        return R, None, None


# LRP-Alpha2Beta1 Rule
class linearlayer_Alpha2Beta1_wrapper_fct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, module, ignorebias):
        #stash module config params and trainable params
        propertynames, values = _configvalues_totensorlist(module)
        weight = module.weight.data.clone()
        if module.bias is None:
          bias = None
        else:
          bias = module.bias.data.clone()
        ignorebiasTensor = torch.tensor([ignorebias], dtype = torch.bool, device = module.weight.device)
        ctx.save_for_backward(x, weight, bias, ignorebiasTensor, *values)
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        # input_, weight, bias, epstensor, *values = ctx.saved_tensors
        input_, weight, bias, ignorebiasTensor, *values = ctx.saved_tensors
        paramsdict = tensorlist_todict(values)
        ignorebias = ignorebiasTensor.item()
        
        # set up module weights and bias
        if bias is None:
          module = nn.Linear(**paramsdict, bias = False )
        else:
          module = nn.Linear(**paramsdict, bias = True )
          module.bias = torch.nn.Parameter(bias)
        module.weight = torch.nn.Parameter(weight)

        X = input_.clone().detach().requires_grad_(True)
        relevance_output = grad_output.clone().detach()
        relevance_output = relevance_filter(relevance_output, top_k_percent = LRP_FILTER_TOP_K, double_side = True)

        # Rule
        pnlinear = posneglinear(module)
        nplinear = negposlinear(module)

        def _f(X, module, relevance_output):
          with torch.enable_grad():
            Z = module(X)
          S = safe_divide(relevance_output, Z.clone().detach(), eps0 = EPS, eps = 0)
          Z.backward(S)
          R = X.data * X.grad.data
          return R

        R1 = _f(X, pnlinear, relevance_output)
        R2 = _f(X, nplinear, relevance_output)
        R = R1 * 2 - R2 * 1         # alpha=2, beta=1

        return R, None, None



# # LRP-|Z| Rule: https://link.springer.com/chapter/10.1007/978-3-030-20518-8_24#Sec3
# class linearlayer_absZ_wrapper_fct(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, module, ignorebias):
#         #stash module config params and trainable params
#         propertynames, values = _configvalues_totensorlist(module)
#         weight = module.weight.data.clone()
#         if module.bias is None:
#           bias = None
#         else:
#           bias = module.bias.data.clone()
#         ignorebiasTensor = torch.tensor([ignorebias], dtype = torch.bool, device = module.weight.device)
#         ctx.save_for_backward(x, weight, bias, ignorebiasTensor, *values)
#         return module.forward(x)

#     @staticmethod
#     def backward(ctx, grad_output):
#         # input_, weight, bias, epstensor, *values = ctx.saved_tensors
#         input_, weight, bias, ignorebiasTensor, *values = ctx.saved_tensors
#         paramsdict = tensorlist_todict(values)
#         ignorebias = ignorebiasTensor.item()
        
#         # set up module weights and bias
#         if bias is None:
#           module = nn.Linear(**paramsdict, bias = False )
#         else:
#           module = nn.Linear(**paramsdict, bias = True )
#           module.bias = torch.nn.Parameter(bias)
#         module.weight = torch.nn.Parameter(weight)

#         X = input_.clone().detach().requires_grad_(True)

#         relevance_output = grad_output.clone().detach()

#         # Rule
#         pnlinear = posneglinear(module, ignorebias = ignorebias)
#         nplinear = negposlinear(module, ignorebias = ignorebias)

#         def _f(X, module, relevance_output):
#           with torch.enable_grad():
#             Z = module(X)
#           S = safe_divide(relevance_output, Z.clone().detach(), eps0 = EPS, eps = 0)
#           Z.backward(S)
#           R = X.data * X.grad.data
#           return R

#         R1 = _f(X, pnlinear, relevance_output)
#         R2 = _f(X, nplinear, relevance_output)
#         R = R1 * 1 + R2 * 1         # alpha=1, beta=-1

#         return R, None, None


class linearlayer_abs_wrapper_fct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, module):
        propertynames, values = _configvalues_totensorlist(module)
        weight = module.weight.data.clone()
        if module.bias is None:
            bias = None
        else:
            bias = module.bias.data.clone()

        ctx.save_for_backward(x, weight, bias, *values)
        return module.forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight, bias, *values = ctx.saved_tensors
        paramsdict = tensorlist_todict(values)

        # set up module weights and bias
        if bias is None:
          module = nn.Linear(**paramsdict, bias = False )
        else:
          module = nn.Linear(**paramsdict, bias = True )
          module.bias = torch.nn.Parameter(bias)
        module.weight = torch.nn.Parameter(weight)
        
        X = input_.clone().detach().requires_grad_(True)
        relevance_output = grad_output.clone().detach()

        with torch.enable_grad():
            Z = module(X) # h
            
            # absLRP: S = R_out / (|Z| + eps)
            S = relevance_output / (Z.abs() + EPS)

            # absLRP numerator virtual forward pass (Equation 12 Numerator)
            # Goal is to compute gradient of (xi * wij)^+
            
            # ha = module_abs(input_abs) = |x| * |w|
            W_abs = weight.abs()
            B_abs = bias.abs() if bias is not None else None
            Z_abs = F.linear(X.abs(), W_abs, B_abs) # ha
            
            # Z_pos: positive contribution part
            Z_pos = (Z + Z_abs)
            
            grads = torch.autograd.grad(outputs=Z_pos, inputs=X, grad_outputs=S)[0]
            R = X.data * grads.data
            print('AbsLRP rel backward: ', R.min().item(), R.max().item())

        return R, None, None







#######################################################
#######################################################
#######################################################

def relevance_filter(r: torch.tensor, top_k_percent: float = 1.0, double_side=False) -> torch.tensor:
    """
    Adopted from: https://github.com/kaifishr/PyTorchRelevancePropagation

    Filter that allows largest k percent values to pass for each batch dimension.
    Filter keeps k% of the largest tensor elements. Other tensor elements are set to
    zero. Here, k = 1 means that all relevance scores are passed on to the next layer.
    Args:
        r: Tensor holding relevance scores of current layer.
        top_k_percent: Proportion of top k values that is passed on.
    Returns:
        Tensor of same shape as input tensor.
    """
    assert 0.0 < top_k_percent <= 1.0

    if top_k_percent < 1.0:
      # size = r.size()
      num_elements = r.size(-1)
      k = max(1, int(top_k_percent * num_elements))
      
      r_filtered = torch.zeros_like(r)
      top_k = torch.topk(input = r, k = k, dim = -1)
      r_filtered.scatter_(dim = -1, index = top_k.indices, src = top_k.values)      

      if double_side:
          bottom_k = torch.topk(r, k=k, dim=-1, largest=False)
          r_filtered.scatter_(-1, bottom_k.indices, bottom_k.values)
          
    return r_filtered


#######################################################
#######################################################
#######################################################
def _configvalues_totensorlist(module):
  propertynames=['in_features', 'out_features']
  values=[]
  for attr in propertynames:
    v = getattr(module, attr)
    # convert it into tensor
    # has no treatment for booleans yet
    if isinstance(v, int):
      v =  torch.tensor([v], dtype = torch.int32, device = module.weight.device) 
    elif isinstance(v, tuple):
      # FAILMODE: if it is not a tuple of ints but e.g. a tuple of floats, or a tuple of a tuple
      v =  torch.tensor(v, dtype = torch.int32, device = module.weight.device)     
    else:
      print('v is neither int nor tuple. unexpected')
      exit()
    values.append(v)

  return propertynames, values


# reconstruct dictionary of config parameters
def tensorlist_todict(values): 
  propertynames=['in_features', 'out_features']
  # idea: paramsdict = { n: values[i]  for i,n in enumerate(propertynames)  } # but needs to turn tensors to ints or tuples!
  paramsdict = {}
  for i, n in enumerate(propertynames):
    v = values[i]
    if v.numel == 1:
        paramsdict[n] = v.item()
    else:
        alist = v.tolist()
        if len(alist) == 1:
          paramsdict[n] = alist[0]
        else:
          paramsdict[n] = tuple(alist)
  return paramsdict


class poslinear(nn.Module):
    def _clone_module(self, module):
        clone = nn.Linear(**{attr: getattr(module, attr) for attr in ['in_features','out_features']})
        return clone.to(module.weight.device)

    def __init__(self, linear, ignorebias):
      super(poslinear, self).__init__()

      self.alinear = self._clone_module(linear)
      self.alinear.weight = torch.nn.Parameter(linear.weight.data.clone().clamp(min = 0)).to(linear.weight.device)
      if ignorebias ==True:
        self.b = None
      else:
          if linear.bias is not None:
              self.b = torch.nn.Parameter(linear.bias.data.clone().clamp(min = 0) )

    def forward(self, x):
        wx = self.alinear(x)
        if self.b is not None:
          v = wx + self.b 
        else:
          v = wx
        
        return v


class posneglinear(nn.Module):
    def _clone_module(self, module):
        clone = nn.Linear(**{attr: getattr(module, attr) for attr in ['in_features', 'out_features']})
        return clone.to(module.weight.device)

    def __init__(self, linear):
      super(posneglinear, self).__init__()

      self.poslinear = self._clone_module(linear)
      self.poslinear.weight = torch.nn.Parameter(linear.weight.data.clone().clamp(min = 0)).to(linear.weight.device)

      self.neglinear = self._clone_module(linear)
      self.neglinear.weight = torch.nn.Parameter(linear.weight.data.clone().clamp(max = 0)).to(linear.weight.device)

      self.poslinear.bias = None
      self.neglinear.bias = None
      # if ignorebias ==True:
      #   self.poslinear.bias = None
      #   self.neglinear.bias = None
      # else:
      #     if linear.bias is not None:
      #         self.poslinear.bias = torch.nn.Parameter(linear.bias.data.clone().clamp(min = 0) )
      #         self.neglinear.bias = torch.nn.Parameter(linear.bias.data.clone().clamp(max = 0) )

    def forward(self,x):
        vp1 = self.poslinear(torch.clamp(x, min = 0))
        vn1 = self.neglinear(torch.clamp(x, max = 0))
        return vp1 + vn1


class negposlinear(nn.Module):
    def _clone_module(self, module):
        clone = nn.Linear(**{attr: getattr(module, attr) for attr in ['in_features','out_features']})
        return clone.to(module.weight.device)

    def __init__(self, linear):
      super(negposlinear, self).__init__()

      self.poslinear = self._clone_module(linear)
      self.poslinear.weight = torch.nn.Parameter(linear.weight.data.clone().clamp(min = 0)).to(linear.weight.device)

      self.neglinear = self._clone_module(linear)
      self.neglinear.weight = torch.nn.Parameter(linear.weight.data.clone().clamp(max = 0)).to(linear.weight.device)
      
      self.poslinear.bias = None
      self.neglinear.bias = None
      # if ignorebias ==True:
      #   self.poslinear.bias = None
      #   self.neglinear.bias = None
      # else:
      #     if linear.bias is not None:
      #         self.poslinear.bias = torch.nn.Parameter(linear.bias.data.clone().clamp(min = 0) )
      #         self.neglinear.bias = torch.nn.Parameter(linear.bias.data.clone().clamp(max = 0) )

    def forward(self,x):
        vp2 = self.poslinear(torch.clamp(x, max = 0)) #on negatives
        vn2 = self.neglinear(torch.clamp(x, min = 0)) #on positives
        return vp2 + vn2
    





#######################################################
#######################################################
# #base routines
#######################################################
#######################################################

# def safe_divide(numerator, divisor, eps0, eps):
#     return numerator / (divisor + eps0 * (divisor == 0).to(divisor) + eps * divisor.sign() )
def safe_divide(numerator, divisor, eps0, eps):
    return numerator / (divisor + eps0 * (divisor == 0).to(divisor))


def lrp_backward(_input, layer, relevance_output, eps0, eps):
    """
    Performs the LRP backward pass, implemented as standard forward and backward passes.
      relevance_output is the gradient_output of the layer
    
    Steps to formula: https://kaifishr.github.io/2021/12/15/relevance-propagation-pytorch.html
    """
    # print(layer.name)
    relevance_output_data = relevance_output.clone().detach()

    # Step 1:
    # compute the total preactivation mass flowing from all neurons in layer l to neuron j in layer (l+1)
    # aka, a forward pass
    with torch.enable_grad():
        Z = layer(_input)
      
    # Step 2:
    # ensures the contributions of each neuron are put in proportion to the total contribution of all neurons
    S = safe_divide(relevance_output_data, Z.clone().detach(), eps0, eps)

    # Step 3:
    # can be interpreted as a backward pass
    Z.backward(S)

    # Step 4:
    # final contributions of each neuron in layer l to all neurons j in layer (l+1)
    # for each neuron i in layer l, compute the element-wise product
    relevance_input = _input.data * _input.grad.data

    return relevance_input #.detach()

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
def get_lrpwrapperformodule(module, 
                            lrp_params, 
                            lrp_layer2method, 
                            ):

  if isinstance(module, nn.ReLU):
    key='nn.ReLU'
    if key not in lrp_layer2method:
      print(key, "not find in lrp_layer2method")
      raise lrplookupnotfounderror( key, "not find in lrp_layer2method")
    
    autogradfunction = lrp_layer2method[key]()  # relu_wrapper_fct()
    return zeroparam_wrapper_class(module, 
                                   autogradfunction=autogradfunction)


  elif isinstance(module, nn.BatchNorm1d):
    key='nn.BatchNorm1d'
    if key not in lrp_layer2method:
      print(key, "not find in lrp_layer2method")
      raise lrplookupnotfounderror(key, "not find in lrp_layer2method")

    autogradfunction = lrp_layer2method[key]()  # relu_wrapper_fct()
    return zeroparam_wrapper_class(module, 
                                   autogradfunction=autogradfunction)


  elif isinstance(module, nn.Linear):
    key='nn.Linear'
    if key not in lrp_layer2method:
      print(key, "not find in lrp_layer2method")
      raise lrplookupnotfounderror( key, "not find in lrp_layer2method")

    autogradfunction = lrp_layer2method[key]()  # linearlayer_*_wrapper_fct()

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
                                    parameter1 = lrp_params['linear_ignorebias'] )
    elif type(autogradfunction) == linearlayer_Alpha2Beta1_wrapper_fct:
       return oneparam_wrapper_class(module, 
                                    autogradfunction = autogradfunction, 
                                    parameter1 = lrp_params['linear_ignorebias'] )
    elif type(autogradfunction) == linearlayer_absZ_wrapper_fct:
       return oneparam_wrapper_class(module, 
                                    autogradfunction = autogradfunction, 
                                    parameter1 = lrp_params['linear_ignorebias'] )

  else:
    raise lrplookupnotfounderror( "found no dictionary entry in lrp_layer2method for this module name:", module)


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
        relevance_output = grad_output.clone().detach()[0]

        # keeps k% of the largest tensor elements
        relevance_output = relevance_filter(relevance_output, top_k_percent = LRP_FILTER_TOP_K)

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
        relevance_output = grad_output.clone().detach()[0]

        # keeps k% of the largest tensor elements
        relevance_output = relevance_filter(relevance_output, top_k_percent = LRP_FILTER_TOP_K)

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
    def forward(ctx, x, module, ignorebias):
        #stash module config params and trainable params
        propertynames, values = _configvalues_totensorlist(module)

        weight = module.weight.data.clone()
        if module.bias is None:
          bias = None
        else:
          bias = module.bias.data.clone()

        ignorebiasTensor=torch.tensor([ignorebias], dtype = torch.bool, device = module.weight.device)

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
        relevance_output = grad_output.clone().detach()[0]
        
        # keeps k% of the largest tensor elements
        relevance_output = relevance_filter(relevance_output, top_k_percent = LRP_FILTER_TOP_K)

        # Rule
        pnlinear = posneglinear(module, ignorebias = ignorebias)
        nplinear = negposlinear(module, ignorebias = ignorebias)

        def _f(X, module, relevance_output):
          with torch.enable_grad():
            Z = module(X)
          S = safe_divide(relevance_output, Z.clone().detach(), eps0 = EPS, eps = 0)
          Z.backward(S)
          R = X.data * X.grad.data
          return R

        R1 = _f(X, pnlinear, relevance_output)
        R2 = _f(X, nplinear, relevance_output)
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
        relevance_output = grad_output.clone().detach()[0]

        # keeps k% of the largest tensor elements
        relevance_output = relevance_filter(relevance_output, top_k_percent = LRP_FILTER_TOP_K)

        # Rule
        pnlinear = posneglinear(module, ignorebias = ignorebias)
        nplinear = negposlinear(module, ignorebias = ignorebias)

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



# LRP-|Z| Rule: https://link.springer.com/chapter/10.1007/978-3-030-20518-8_24#Sec3
class linearlayer_absZ_wrapper_fct(torch.autograd.Function):
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
        relevance_output = grad_output.clone().detach()[0]

        # keeps k% of the largest tensor elements
        relevance_output = relevance_filter(relevance_output, top_k_percent = LRP_FILTER_TOP_K)

        # Rule
        pnlinear = posneglinear(module, ignorebias = ignorebias)
        nplinear = negposlinear(module, ignorebias = ignorebias)

        def _f(X, module, relevance_output):
          with torch.enable_grad():
            Z = module(X)
          S = safe_divide(relevance_output, Z.clone().detach(), eps0 = EPS, eps = 0)
          Z.backward(S)
          R = X.data * X.grad.data
          return R

        R1 = _f(X, pnlinear, relevance_output)
        R2 = _f(X, nplinear, relevance_output)
        R = R1 * 1 + R2 * 1         # alpha=1, beta=-1

        return R, None, None


#######################################################
#######################################################
#######################################################

def relevance_filter(r: torch.tensor, top_k_percent: float = 1.0) -> torch.tensor:
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
      top_k = torch.topk(input = r, k = k, dim = -1)
      r = torch.zeros_like(r)
      r.scatter_(dim = 0, index = top_k.indices, src = top_k.values)

    return r


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

    def __init__(self, linear, ignorebias):
      super(posneglinear, self).__init__()

      self.poslinear = self._clone_module(linear)
      self.poslinear.weight = torch.nn.Parameter(linear.weight.data.clone().clamp(min = 0)).to(linear.weight.device)

      self.neglinear = self._clone_module(linear)
      self.neglinear.weight = torch.nn.Parameter(linear.weight.data.clone().clamp(max = 0)).to(linear.weight.device)

      if ignorebias ==True:
        self.poslinear.bias = None
        self.neglinear.bias = None
      else:
          if linear.bias is not None:
              self.poslinear.bias = torch.nn.Parameter(linear.bias.data.clone().clamp(min = 0) )
              self.neglinear.bias = torch.nn.Parameter(linear.bias.data.clone().clamp(max = 0) )

    def forward(self,x):
        vp1 = self.poslinear(torch.clamp(x, min = 0))
        vn1 = self.neglinear(torch.clamp(x, max = 0))
        return vp1 + vn1


class negposlinear(nn.Module):
    def _clone_module(self, module):
        clone = nn.Linear(**{attr: getattr(module, attr) for attr in ['in_features','out_features']})
        return clone.to(module.weight.device)

    def __init__(self, linear, ignorebias):
      super(negposlinear, self).__init__()

      self.poslinear = self._clone_module(linear)
      self.poslinear.weight = torch.nn.Parameter(linear.weight.data.clone().clamp(min = 0)).to(linear.weight.device)

      self.neglinear = self._clone_module(linear)
      self.neglinear.weight = torch.nn.Parameter(linear.weight.data.clone().clamp(max = 0)).to(linear.weight.device)

      if ignorebias ==True:
        self.poslinear.bias = None
        self.neglinear.bias = None
      else:
          if linear.bias is not None:
              self.poslinear.bias = torch.nn.Parameter(linear.bias.data.clone().clamp(min = 0) )
              self.neglinear.bias = torch.nn.Parameter(linear.bias.data.clone().clamp(max = 0) )

    def forward(self,x):
        vp2 = self.poslinear(torch.clamp(x, max = 0)) #on negatives
        vn2 = self.neglinear(torch.clamp(x, min = 0)) #on positives
        return vp2 + vn2
    





#######################################################
#######################################################
# #base routines
#######################################################
#######################################################

def safe_divide(numerator, divisor, eps0, eps):
    return numerator / (divisor + eps0 * (divisor == 0).to(divisor) + eps * divisor.sign() )


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

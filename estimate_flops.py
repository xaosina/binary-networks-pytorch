import torch
from millify import millify
from pthflops import count_ops
import bnn.models.resnet as models
  


def get_stats(model, ignore_layers = ['first_conv'], binary_layers=['conv']):
    """
    The get_stats function computes the number of FLOPs and BOPs for a given model.
    It also computes the total memory footprint of all parameters in bytes.
    The function takes two arguments: 
        1) The model to be analyzed, and 
        2) A list of layers that should not be counted as binary operations (e.g., first_conv).
    
        It returns a tuple with three elements: 
            1) The total number of FLOPs, 
            2) The total number of BOPs, and 
            3) Total memory footprint in bytes.
    
    :param model: Specify the model for which the stats are to be calculated
    :param ignore_layers=('first_conv'): Ignore the first convolutional layer of the model
    :param binary_layers=('conv'): Specify which layers are binary
    :return: The number of flops and bops for the model
    """
    
    device = 'cuda:0'
    model.to(device)
    inp = torch.rand(1,3,224,224).to(device)
    
    all_ops, all_data = count_ops(model, inp)
    flops, bops = 0, 0
    for op_name, ops_count in all_data:
        if any(op in op_name for op in binary_layers) and not op_name in ignore_layers:
            bops += ops_count
        else:
            print(op_name, ops_count)
            flops += ops_count

    prefixes = ['k', 'M', 'G']
    flops_str = millify(flops, precision=2, prefixes=prefixes)
    bops_str = millify(bops, precision=2, prefixes=prefixes)
    
    print(f'BOPs: {bops_str}')
    print(f'FLOPs: {flops_str}')
    

    memory_f, memory_b = 0, 0 
    for name, param in model.named_parameters():
        if any(op in name for op in binary_layers) and not name in ignore_layers:
            memory_b+=param.nelement()*param.element_size() 
        else:
            memory_f+=param.nelement()*param.element_size() 
            
            
    m_flops_str = millify(memory_f, precision=2, prefixes=prefixes)
    m_bops_str = millify(memory_b, precision=2, prefixes=prefixes)   

    print(f'B Memory : {m_bops_str}')
    print(f'F Memory : {m_flops_str}')
    
    return {'flops':flops_str, 'bops':bops_str, 'memory_B':m_bops_str, 'memory F':m_flops_str}



if __name__=='__main__':
    from models.group_net import resnet18
    model = resnet18(num_classes=200)
    #model = models.__dict__["resnet18"](stem_type="basic", num_classes=200)
    get_stats(model, ignore_layers = ['first_conv'])



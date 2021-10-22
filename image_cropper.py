import torch
def crop_img(source_tensor, target_tensor):
    source_tensor_size = source_tensor.size()[2]
    target_tensor_size = target_tensor.size()[2]
    diff = (source_tensor_size - target_tensor_size) // 2 # up and down strip --> so //2 takes only one side
    return source_tensor[:,:,diff:-diff,diff:-diff]

if __name__=="__main__":
    src = torch.rand(4,128,280,280)
    target = torch.rand(4,256,200,200)
    crop_tensor = crop_img(src,target)
    print(crop_tensor.size())

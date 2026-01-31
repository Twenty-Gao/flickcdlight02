from model.decoder import FlickCD
from calflops import calculate_flops

model = FlickCD((8, 8, 16),(4,4,8), load_pretrained=False)
model = model.cuda()

flops,macs,params = calculate_flops(model, input_shape=(1,3,256,256), output_as_string=True, output_precision=4)
print("FLOPs:%s     MACs:%s     Params:%s \n" %(flops, macs, params))


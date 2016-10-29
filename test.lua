require 'torch'
require 'xlua'
require 'paths'
require 'image'
local py = require 'fb.python'

require 'transform'
optnet = require 'optnet'

-- Load optional libraries
xrequire('cunn')
xrequire('cudnn')

torch.setdefaulttensortype('torch.FloatTensor')
opts_mem = {inplace=true, reuseBuffers=true, mode='inference'}

local options = require 'options'
local data = require 'data'

local opts = options.parse(arg)

local activThresh = 0.003

data.checkIntegrity(opts)

-- Load the model
local model = nil
if opts.dataset == 'mpii' then
  model = torch.load('models/human_pose_mpii.t7')
else 
  model = torch.load('models/human_pose_lsp.t7')
end

if opts.useGPU then 
  if opts.usecudnn then
    cudnn.fastest = true
    cudnn.convert(model, cudnn)
  end
  model = model:cuda()
end

if opts.useGPU then
	optnet.optimizeMemory(model, torch.zeros(1,3,opts.res,opts.res):cuda(), opts_mem)
else
	optnet.optimizeMemory(model, torch.zeros(1,3,opts.res,opts.res), opts_mem)
end

model:evaluate()

-- Import python libraries and set pairs
py.exec([=[
import re
import os
import numpy as np
import matplotlib.pyplot as plt
pairs = np.array([[1,2], [2,3], [3,7], [4,5], [4,7], [5,6], [7,9], [9,10], [14,9], [11,12], [12,13], [13,9], [14,15], [15,16]])-1

def load_img_path_list(path):
    """

    :param path: the test img folder
    :return:
    """
    p = re.compile(".*extract.jpg")
    list_path = os.listdir(path)
    # change to reg to match extension
    result = ["%s/%s" % (path, x) for x in list_path if p.match(x)]
    return result, len(result)

def write_joint(img_filename, joints):
    filename = ''.join(img_filename.split('.')[:-1])
    filename = "%s_joint.txt" % filename
    f = open(filename,'w')
    for j in joints[0]:
        f.write("%d\t%d\n" % (j[0], j[1]))
    f.close()
]=])


img_path = {}

-- count, the image count
n = 0

--scan the dir
dir = opts.img_folder
print(dir)
res = py.eval('load_img_path_list(a)', {a = dir})
img_path = res[1]
n = res[2]
local predictions = torch.Tensor(n,16,2)

-- Set the progress bar
xlua.progress(0,n)

for i=1,n  do
  -- Load and prepare the data
  local img = nil

  img = image.load(img_path[i])
  local center = (function() return torch.Tensor({img:size()[3]/2,img:size()[2]/2}) end)() 
  local scale = (function() return 0.89 end)() 
  local input = crop(img,center,scale,opts.res)
  input = (function() if opts.useGPU then return input:cuda() else return input end end)()
  
  -- Do the forward pass and get the predicitons
  local output = model:forward(input:view(1,3,opts.res,opts.res))
  
  output = applyFn(function (x) return x:clone() end, output)
  local flippedOut = nil
  if opts.useGPU then
        flippedOut = model:forward(flip(input:view(1,3,opts.res,opts.res):cuda()))
  else
        flippedOut = model:forward(flip(input:view(1,3,opts.res,opts.res)))
  end
  flippedOut = applyFn(function (x) return flip(shuffleLR(x)) end, flippedOut)
  output = applyFn(function (x,y) return x:add(y):div(2) end, output, flippedOut):float()
	
  output[output:lt(0)] = 0
  xlua.progress(i,n)
  
  local preds_hm, preds_img = getPreds(output[1], center, scale)

  if not opts.eval then
    py.exec("write_joint(name, joint)", {name=img_path[i],joint=preds_hm})
  else
    predictions[{{i},{},{}}] = preds_img
  end

  collectgarbage()
end

import torch.nn as nn
import torch

class ROIPoolling(nn.Module):
    def __init__(self, levels=[1]):
        super().__init__()
        
        self.levels = levels
        
    
    

    def forward(self, features, roiss):
        
        image_extraction = [] 

        for indix, rois in enumerate(roiss):
         
        
          level_res = []

        

            
          feat = features[indix]
          h, w = feat.shape[1:]

          normalized_rois = rois / torch.tensor((1024, 800, 1024, 800), device = rois.device)
          
          n = len(normalized_rois)
          x1 = normalized_rois[:,0]
          y1 = normalized_rois[:,1]
          x2 = normalized_rois[:,2]
          y2 = normalized_rois[:,3]

          x1 = torch.clamp( torch.floor(x1 * w).type(torch.int), 0)
          x2 = torch.clamp( torch.ceil(x2 * w).type(torch.int), 0)
          
          y1 = torch.clamp( torch.floor(y1 * h).type(torch.int), 0)
          y2 = torch.clamp( torch.ceil(y2 * h).type(torch.int), 0)
          
          
          for output_shape in self.levels:
            output_shape = (output_shape, output_shape)
            maxpool = nn.AdaptiveMaxPool2d(output_shape)
            
            for i in range(n):
      
                

                if x1[i] == 0 and x2[i] == 0:
                  x2[i] = x2[i]+1
                if y1[i] == 0 and y2[i] == 0:
                  y2[i] = y2[i]+1

                if x1[i] >= w:
                  x1[i] = torch.tensor((w-1)).type(torch.int)
                if y1[i] >= h:
                  y1[i] = torch.tensor((h-1)).type(torch.int)

                
                img = maxpool(feat[ :, y1[i]:y2[i], x1[i]:x2[i]])
                level_res.append(img)
          
          image_extraction.append(torch.stack(level_res,dim=0).view( len(rois) , -1))
          

        
         
     
        return image_extraction

        
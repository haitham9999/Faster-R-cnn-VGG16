import torch.nn as nn
import torch
import torch.nn.functional as F

class rpn_head(nn.Module):
    
    def __init__(self, in_channels=512, mid_channels=512, num_anchors = 9):
        super(rpn_head, self).__init__()   

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
                    
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)

        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1 )
              
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
              torch.nn.init.normal_(layer.weight, std=0.01)
              torch.nn.init.constant_(layer.bias, 0)


    def forward(self,  feature_map):
        
        x = F.relu(self.conv(feature_map))
        pred_anchor_locs_all = self.bbox_pred(x).permute(0,2,3,1).contiguous().view(len(feature_map), -1, 4)
        pred_cls_scores_all  = self.cls_logits(x).permute(0,2,3,1).contiguous().view(len(feature_map), -1, 1)

		
        return {'pred_cls_scores_all':pred_cls_scores_all, 'pred_anchor_locs_all':pred_anchor_locs_all }


# equal_opportunity.py

import torch
from hal.metrics import Metric
from typing import Any, Callable, Optional

__all__ = ['EqualOpportunity']

class EqualOpportunity(Metric):
    def __init__(self,
                 compute_on_step: bool = False,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None,
                 ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("pred0", default=[],dist_reduce_fx=None)
        self.add_state("pred1", default=[], dist_reduce_fx=None)

    def update(self, yhat, s, y):
        pred = yhat.data.max(1)[1]
        s = s.long()
        s_cond = s[y==1].squeeze().long()
        pred_cond = pred[y==1]

        self.pred0.append(pred_cond[s_cond==0])
        self.pred1.append(pred_cond[s_cond==1])

    def compute(self):
        pred0 = torch.cat(self.pred0, dim=0)
        pred1 = torch.cat(self.pred1, dim=0)

        if len(pred0) > 0 and len(pred1) > 0:
            deo = len(pred1[pred1==1])/len(pred1) - len(pred0[pred0==1])/len(pred0)
        else:
            deo = 1
        return abs(deo)
import torch
import torch.nn as nn

class UNI(nn.Module):
    def __init__(self, args):
        super(UNI, self).__init__()
        _, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.use_a = False
        if args["uni"]["modal"] == "a":
            self.d_in = args["uni"].get("d_in", 768)
            self.use_a = True
        else: # visual component
            self.d_in = args["uni"].get("d_in", 768)
        self.mean_before = args["uni"].get("mean_before", False)
        print(f" -------- Using MEAN before -----------") if self.mean_before else None
        self.BN_in = nn.BatchNorm1d(self.d_in) if self.mean_before else None
        self.dropout = nn.Dropout(p=0.1)
        # self.dropout = nn.Dropout(p=0.0)
        self.activ = nn.ReLU()
        self.d_ffw = args["uni"]["d_ffw"]
        # projector
        # import pdb; pdb.set_trace()
        # self.ffw = nn.Linear(self.d_in, self.d_ffw)
        # self.LN_out = nn.LayerNorm(self.d_ffw)
        self.ffw = nn.Linear(self.d_in, self.d_ffw)
        # Clf
        self.clf = nn.Linear(self.d_ffw, 1)


    def forward(self, x_l, x_a, x_v): # for compatibility with wrapper
        """x: (B, L, D)
        """
        if self.use_a:
            x = x_a
        else:
            x = x_v

        if self.mean_before:
            x = self.dropout(self.BN_in(torch.mean(x, dim=1)))
            # x = torch.mean(x, dim=1)
            # x = self.activ(self.ffw(x))
            # x = self.activ(self.ffw(x))
            x = self.activ(self.ffw(x))
        else:
            # projection
            x = torch.mean(
                self.activ(self.ffw(x)),
                dim=1
            ).squeeze(1) # (B, d_ffw)
        # clf
        return self.clf(x) # (B, 1)


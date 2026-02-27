import torch
import math
from fast_krr.opts.optimizer import Optimizer
from fast_krr.opts.utils.general import _get_leverage_scores
from fast_krr.opts.utils.general import _get_L, _apply_precond
from fast_krr.opts.utils.bcd import (
    _get_block,
    _get_block_properties,
)

#also returns a sum of the squared error
def _get_block_update_w_err(model, w, block, precond):
    
    # Compute the block gradient
    gb = model._get_block_grad(w, block)
    resids=gb-model.lambd * model.w[block]
    #resids=gb
    
    # Apply the preconditioner
    dir = _apply_precond(gb, precond)
    return dir, (resids**2).sum()

class ASkotchV3(Optimizer):
    def __init__(
        self,
        model,
        block_sz,
        sampling_method="uniform",
        precond_params=None,
        eta=None,
        p=None,
        accelerated=True,
        rho_stop=1e-4,
    ):
        super().__init__(model, precond_params)
        self.block_sz = block_sz
        self.eta = eta if eta is not None else 4*self.block_sz / self.model.n
        self.p = p if p is not None else 100
        self.accelerated = accelerated

        # TODO(pratik): check that nu > mu and mu * nu <= 1

        # Compute sampling probabilities
        if sampling_method == "rls":
            leverage_scores = _get_leverage_scores(
                model=self.model,
                size_final=int(self.model.n**0.5),
                lam_final=self.model.lambd,
                rls_oversample_param=5,
            )
            self.probs = leverage_scores / torch.sum(leverage_scores)
        elif sampling_method == "uniform":
            self.probs = torch.ones(self.model.n) / self.model.n
        self.probs_cpu = self.probs.cpu().numpy()
        self.i = 0

        if self.accelerated:
            self.dist_new = 0.0
            self.dist_old = 0.0
            #rho=1 means no acceleration, only start to accelerate later
            self.rho = 0.01
            self.m_old = torch.zeros(self.model.n,device=self.model.device)
            self.m_new = torch.zeros(self.model.n,device=self.model.device)
            self.temp = torch.zeros(self.model.n,device=self.model.device)
            self.ratio = 0
            self.rho_stop = rho_stop 
            self.stopped = False


    def step(self):
        # Randomly select block_sz distinct indices
        if self.rho<self.rho_stop:
            self.stopped=True
            return
        block = _get_block(self.probs, self.probs_cpu, self.block_sz)

        # Compute block preconditioner and learning rate
        block_precond, block_eta, _ = _get_block_properties(
            self.model, self.precond_params, [block], False
        )
        block_precond = block_precond[0]
        block_eta = block_eta[0]

        # Get the update direction
        dir,sum_o_sqrerr = _get_block_update_w_err(self.model, self.model.w, block, block_precond)

        if self.accelerated:
            self.temp[block] += block_eta * dir 
            
            self.m_new = (1-self.rho)/(1+self.rho)*(self.m_old-self.temp)
            self.model.w = self.model.w - self.temp + self.eta*self.m_new
            self.temp[:]=0
            self.m_old = self.m_new.clone()
            self.dist_new += sum_o_sqrerr
            if self.i%self.p==self.p-1:
                cnt=(self.i+1)//self.p
                a_old = cnt**math.log(cnt)
                a_new = (cnt+1)**math.log(cnt+1)
                if cnt>=2:
                    if cnt==2:
                        self.ratio = self.dist_new / self.dist_old
                    else:
                        self.ratio = self.ratio*(a_old/a_new) + self.dist_new / self.dist_old * (1 - a_old/a_new)
                    self.rho = max(0,1 - self.ratio**(1/self.p))
                self.dist_old=self.dist_new
                self.dist_new=0
        else:
            self.model.w[block] -= block_eta * dir
        self.i+=1

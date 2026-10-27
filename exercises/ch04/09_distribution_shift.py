# -*- coding: utf-8 -*-
"""4.9综合练习：训练域分类器，再做协变量加权回归；八处TODO。"""
import math
import torch
from torch.nn import functional as F


def fit_domain(source,target):
    domain_x=None  # TODO 1：拼接两个域的特征，源在前。
    domain_y=None  # TODO 2：源域标签0，目标域标签1；使用float64。
    if domain_x is None or domain_y is None:
        print('请集中填写八处TODO，接通域检测和加权训练。');return None
    a=torch.zeros((),dtype=torch.float64,requires_grad=True)
    b=torch.zeros((),dtype=torch.float64,requires_grad=True)
    for _ in range(400):
        logits=a*domain_x+b
        loss=None  # TODO 3：域分类的平均BCE，直接接logits。
        if loss is None:return None
        pass  # TODO 4：反向传播域分类损失。
        if a.grad is None or b.grad is None:
            print('请先计算域分类梯度。');return None
        with torch.no_grad():a-=.1*a.grad;b-=.1*b.grad
        a.grad=None;b.grad=None
    return a.detach(),b.detach()


def main():
    torch.set_num_threads(1)
    rng=torch.Generator().manual_seed(4910)
    sample=lambda n,mean:torch.randn(n,generator=rng,dtype=torch.float64)+mean
    source_domain=sample(400,-.6);target_adapt=sample(200,.6)
    x=sample(600,-.6);y=x.square()+.1*sample(600,0.)
    target_x=sample(600,.6);target_y=target_x.square()+.1*sample(600,0.)
    domain=fit_domain(source_domain,target_adapt)
    if domain is None:return
    a,b=domain
    log_ratio=None  # TODO 5：域logits加域样本数量比例的对数修正。
    if log_ratio is None:
        print('请补上密度比中的域先验修正。');return
    weight=None  # TODO 6：先截断log_ratio到log(10)，再取exp并detach。
    if weight is None:return
    print('域参数a/b、域先验修正倍数:',a.item(),b.item(),len(source_domain)/len(target_adapt))
    print('权重均值/最大值/ESS:',weight.mean().item(),weight.max().item(),
          (weight.sum().square()/weight.square().sum()).item())
    design=torch.stack((x,torch.ones_like(x)),dim=1)
    results=[]
    for method,w in (('unweighted',torch.ones_like(weight)),('corrected',weight)):
        theta=torch.zeros(2,dtype=torch.float64,requires_grad=True)
        for _ in range(300):
            prediction=design@theta
            per_sample=None  # TODO 7：每个样本的半平方误差，先不汇总。
            if per_sample is None:return
            objective=None  # TODO 8：权重乘逐样本损失，再按样本数平均。
            if objective is None:return
            objective.backward()
            with torch.no_grad():theta-=.03*theta.grad
            theta.grad=None
        results.append((method,theta.detach().clone()))
    # 所有方案固定后才使用目标评估标签，不把它传给域分类器。
    for method,theta in results:
        with torch.no_grad():
            source_loss=((design@theta-y).square().mean()/2).item()
            target_loss=((theta[0]*target_x+theta[1]-target_y).square().mean()/2).item()
        print('方案/参数/源域与目标域半MSE:',method,theta.tolist(),source_loss,target_loss)
    print('解释：域分类标签和任务标签分别是什么？为什么这里不能遗漏2倍修正？')
    print('解释：加权的假设是什么？如果只有标签规则翻转，域检测器能发现吗？')


if __name__=='__main__':main()

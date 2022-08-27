from torch import optim

class Optimizer(object):
    def __init__(self, params, config, coef=None):
        if not (isinstance(params, list) or isinstance(params, tuple)):
            params = [params]

        if coef is None:
            coef = [1.0] * len(params)
        else:
            if isinstance(coef, list) or isinstance(coef, tuple):
                assert len(coef) == len(params)
            else:
                coef = [coef] * len(params)
        self.coef = coef

        param_groups = []
        for param in params:
            param_groups.append({"params": param, "lr": 0})

        opt = config["optimizer"].lower()
        assert opt == "sgd" or opt == "adam"
        if opt == "sgd":
            self.opt = optim.SGD(
                param_groups, momentum=config["momentum"], weight_decay=config["wd"]
            )
        elif opt == "adam":
            self.opt = optim.Adam(param_groups, weight_decay=0)

        self.lr_func = eval(config["lr_func"])

        if "clip_grads" in config:
            self.clip_grads = config["clip_grads"]
            self.clip_low = config["clip_low"]
            self.clip_high = config["clip_high"]
        else:
            self.clip_grads = False

    def zero_grad(self):
        self.opt.zero_grad()

    def step(self, epoch):
        if self.clip_grads:
            self.clip()

        lr = self.lr_func(epoch)
        for i, param_group in enumerate(self.opt.param_groups):
            param_group["lr"] = lr * self.coef[i]
        self.opt.step()
        return lr

    def clip(self):
        low, high = self.clip_low, self.clip_high
        params = []
        for param_group in self.opt.param_groups:
            params += list(filter(lambda p: p.grad is not None, param_group["params"]))
        for p in params:
            mask = p.grad.data < low
            p.grad.data[mask] = low
            mask = p.grad.data > high
            p.grad.data[mask] = high

    def load_state_dict(self, opt_state):
        self.opt.load_state_dict(opt_state)

class StepLR:
    def __init__(self, lr, lr_epochs):
        assert len(lr) - len(lr_epochs) == 1
        self.lr = lr
        self.lr_epochs = lr_epochs

    def __call__(self, epoch):
        idx = 0
        for lr_epoch in self.lr_epochs:
            if epoch < lr_epoch:
                break
            idx += 1
        return self.lr[idx]
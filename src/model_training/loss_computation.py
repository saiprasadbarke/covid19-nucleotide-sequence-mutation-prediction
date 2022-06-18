class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, scheduler=None, optimizer=None):
        self.generator = generator
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def __call__(self, x, y):
        x = self.generator(x)
        log_probs = x.contiguous().view(-1, x.size(-1))
        targets = y.contiguous().view(-1).long()
        loss = self.criterion(log_probs, targets)

        if self.optimizer is not None:
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()

        return loss.data.item()

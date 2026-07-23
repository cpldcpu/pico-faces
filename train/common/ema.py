import torch


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone().float() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                s.copy_(v)

    def state_dict(self):
        return self.shadow

    @torch.no_grad()
    def load_state_dict(self, shadow):
        """Overwrite the shadow from a saved EMA (e.g. resuming a finetune)."""
        for k, v in shadow.items():
            if k in self.shadow:
                self.shadow[k].copy_(v.to(self.shadow[k].device).float())

    def copy_to(self, model):
        model.load_state_dict({k: v for k, v in self.shadow.items()}, strict=True)

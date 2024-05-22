
import torch

class SampleTimesAndLabels:

    def __init__(self):
        pass


    def _sample_times_and_labels(self, durations, event_observeds, horizon):
        raise NotImplementedError()


    def __call__(self, durations, event_observeds, horizon):
        return self._sample_times_and_labels(durations, event_observeds, horizon)

class SampleTimesAndLabelsDelta(SampleTimesAndLabels):

    def _sample_times_and_labels(self, durations, event_observeds, horizon):
        ts = durations.clone().view(len(durations), 1)
        alives = 1 - event_observeds.clone().type(ts.dtype)

        assert ts.shape == (len(durations), 1), f"{ts.shape=} has a bad shape."
        assert durations.shape == ts.shape, f"Different shapes. {durations.shape=} != {ts.shape=}"
        assert alives.shape == ts.shape, f"Different shapes. {alives.shape=} != {ts.shape=}"
        return ts, alives


class SampleTimesAndLabelsGaussianDelta(SampleTimesAndLabels):

    def __init__(self, σ):
        self._σ = σ


    def _sample_times_and_labels(self, durations, event_observeds, horizon):
        η = torch.randn(len(durations), 1) * self._σ
        ts = durations + η

        ts[~event_observeds] = durations[~event_observeds]
        ts = ts.clamp(0)

        alives = (ts < durations) | ~event_observeds

        assert durations.shape == ts.shape
        return ts, alives.type(ts.dtype)

class WeightedBCELoss:

    def __init__(self, σ_gaussian_delta, weight=1, label_smoothing=1e-3):
        super().__init__()
        self.my_weight = weight
        self.label_smoothing = label_smoothing

        self.sample_times_and_labels = SampleTimesAndLabelsGaussianDelta(σ=σ_gaussian_delta)

        assert 0 <= self.my_weight <= 1, f"{self.my_weight=} is not in the interval [0, 1]."


    def _get_weights(self, targets):
        targets_binary = targets > .5
        weight = torch.ones_like(targets) * (1 - self.my_weight)
        weight[~targets_binary] = self.my_weight
        assert torch.all((0 <= weight) & (weight <= 1)), "BCE weights outside [0, 1]."
        return weight


    def __call__(self, outputs, targets, ts):
        targets = torch.clamp(targets, self.label_smoothing, 1 - self.label_smoothing)
        outputs.data.clamp_(self.label_smoothing, 1 - self.label_smoothing)

        weight = self._get_weights(targets)

        loss = torch.nn.functional.binary_cross_entropy(
            outputs,
            targets,
            weight=weight,
            size_average=None,
            reduce=None,
            reduction='mean'
        )

        assert torch.all(-1e-2 <= outputs), f"Too small values in outputs. {outputs[0 > outputs]=}"
        assert torch.all(outputs <= 1), "Too large values in outputs."
        return loss


class SuMoLoss:

    def __init__(self, weight=1, label_smoothing=1e-3):
        super().__init__()
        self.my_weight = weight
        self.label_smoothing = label_smoothing

        self.sample_times_and_labels = SampleTimesAndLabelsDelta()


    def _get_δS(self, S_t, ts):
        grads = torch.autograd.grad(
            outputs=S_t,
            inputs=ts,
            grad_outputs=torch.ones_like(S_t),
            create_graph=True,
            retain_graph=True
        )[0]
        return grads


    def _get_f(self, S_t, ts):
        grads = self._get_δS(S_t, ts)

        f = -grads

        f[torch.isnan(f) & (S_t < 1e-3)] = 0.

        assert grads.shape == S_t.shape, f"Shapes don't match: {grads.shape=} {S_t.shape=}"
        assert not torch.any(torch.isnan(f)), f"f has NaNs, {f[torch.isnan(f)]=}, {S_t[torch.isnan(f)]=}, {ts[torch.isnan(f)]=}"
        assert torch.all(f >= -0.1), f"f is negative. {f[f < 0]=}, {S_t[f < 0]=}"
        return f


    def _get_f_ll(self, f, alives, ε=1e-16):
        alives_binary = alives > 0.5
        f_ll = torch.log(f[~alives_binary].clamp(ε)).clamp(-10).sum() * self.my_weight
        return f_ll


    def _get_S_ll(self, S, alives, ε=1e-16):
        alives_binary = alives > 0.5
        S_ll = torch.log(S[alives_binary].clamp(ε)).clamp(-10).sum()
        return S_ll


    def __call__(self, outputs, alives, ts):
        f = self._get_f(outputs, ts)
        S = outputs

        f_ll = self._get_f_ll(f, alives)
        S_ll = self._get_S_ll(f, alives)

        loss = -(f_ll + S_ll) / len(outputs)

        assert not torch.any(torch.isnan(S)), f"Found NaN in outputs. {f[torch.isnan(S)]=}, {S_t[torch.isnan(S)]=}, {ts[torch.isnan(S)]=}"
        assert torch.all(-1e-2 <= S), f"Too small values in outputs. {f[S_t < 0]=}, {S_t[S_t < 0]=}"
        assert torch.all(S <= 1), "Too large values in outputs."
        return loss
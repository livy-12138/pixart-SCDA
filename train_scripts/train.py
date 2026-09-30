import argparse
import csv
import datetime
import json
import os
import sys
import time
import types
import warnings
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from diffusers.models import AutoencoderKL
try:
    from mmcv.runner import LogBuffer
except ImportError:
    class LogBuffer:
        def __init__(self): self.output = {}; self._values = {}
        def update(self, values, count=1):
            for key, value in values.items(): self._values.setdefault(key, []).append(float(value))
        def average(self, *args, **kwargs):
            self.output = {key: sum(values) / len(values) for key, values in self._values.items() if values}
        def clear(self): self.output = {}; self._values = {}
from torch.utils.data import RandomSampler

from diffusion import IDDPM
from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.model.builder import build_model
from diffusion.utils.checkpoint import save_checkpoint, load_checkpoint
from diffusion.utils.data_sampler import AspectRatioBatchSampler, BalancedAspectRatioBatchSampler
from diffusion.utils.dist_utils import get_world_size, clip_grad_norm_
from diffusion.utils.logger import get_root_logger
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusion.utils.optimizer import build_optimizer, auto_scale_lr

warnings.filterwarnings("ignore")  # ignore warning

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

# Optional frozen PixArt teacher used by the conservative semantic-distillation
# experiment.  It is kept outside the optimizer and never receives gradients.
teacher_model = None


def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = 'true'
    os.environ["FSDP_AUTO_WRAP_POLICY"] = 'TRANSFORMER_BASED_WRAP'
    os.environ["FSDP_BACKWARD_PREFETCH"] = 'BACKWARD_PRE'
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = 'PixArtBlock'


def ema_update(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


def configure_trainable_parameters(model: nn.Module, train_semantic_only: bool,
                                   trainable_contains=None):
    """Freeze PixArt and train only the semantic-conditioning adapters.

    `trainable_contains` replaces the adapter list with a substring filter, for
    mechanisms that introduce no parameters of their own and must instead adapt
    a named part of the backbone -- pair replacement rewrites the DiT's own
    cross-attention, so that is the part it trains.
    """
    if trainable_contains:
        needles = tuple(trainable_contains)
        trainable = []
        for name, parameter in model.named_parameters():
            parameter.requires_grad = any(needle in name for needle in needles)
            if parameter.requires_grad:
                trainable.append(name)
        if not trainable:
            raise ValueError(f'trainable_contains={needles} matched no parameters')
        logger.info('Substring-restricted training enabled: %d parameter tensors (%s)',
                    len(trainable), ', '.join(needles))
        return
    if not train_semantic_only:
        return
    trainable = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(
            ('semantic_adapters.', 'semantic_layer_scale', 'semantic_time_gate.',
             'semantic_token_cross_attention.', 'semantic_token_gate',
             'semantic_token_layer_gate')
        )
        if parameter.requires_grad:
            trainable.append(name)
    if not trainable:
        raise ValueError('train_semantic_only=True but semantic conditioning is disabled')
    logger.info(f'Semantic-only training enabled: {len(trainable)} parameter tensors')


# The two scalars that control how strongly the token-pair branch is injected.
# They are optimised with their own learning rate; see GATE_PARAMETER_NAMES and
# results/GATE_OPTIMIZATION.md.
GATE_PARAMETER_NAMES = ('semantic_token_layer_gate', 'semantic_token_gate')


def set_gate_requires_grad(model, flag):
    for name, parameter in model.named_parameters():
        if name in GATE_PARAMETER_NAMES:
            parameter.requires_grad = bool(flag)


def split_gate_parameters(model):
    """(gate_params, other_params) among the currently trainable parameters."""
    gate, other = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (gate if name in GATE_PARAMETER_NAMES else other).append(parameter)
    return gate, other


def all_grads_finite(model):
    """True if every existing gradient is free of inf/NaN.

    Must be checked *before* ``clip_grad_norm_``: that call invokes
    ``scaler.unscale_()``, and calling ``unscale_()`` twice with no intervening
    ``scaler.step()`` raises ``RuntimeError``.  Skipping a batch by clipping
    first and discarding afterwards therefore crashes on the following step.
    """
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            return False
    return True


def parameter_grad_norm(model, names):
    """L2 norm of the gradients of the named parameters (0 if none)."""
    total = None
    for name, parameter in model.named_parameters():
        if name not in names or parameter.grad is None:
            continue
        squared = parameter.grad.detach().float().pow(2).sum()
        total = squared if total is None else total + squared
    if total is None:
        return 0.0
    return float(total.sqrt())


class GateTrainingSchedule:
    """Everything that lets the TP-SCDA gate actually learn.

    Rationale (measured, see results/GATE_OPTIMIZATION.md): the gate used to sit
    in the saturated tail of the sigmoid where ``dlambda/dw ~= 0.018``, and the
    113 gate scalars shared the 3.5e-6 learning rate of the 8M-parameter
    projections, so they could travel at most ~0.05 in logit space over 14 786
    steps while ~1.0 was needed.  This class applies, per step:

      * a warm-up ramp of the injection ceiling (0 -> target), so early training
        is not perturbed while the projections are still random;
      * a freeze window (`semantic_gate_freeze_steps`) so the gates do not chase
        noise gradients from an untrained projection;
      * annealing of the semantic L2 coefficient (initial -> final), which is a
        constant one-directional pressure towards zero injection.

    The distillation anchor is deliberately NOT touched here.
    """

    def __init__(self, model, config, log):
        self.model = model
        self.log = log
        self.freeze_steps = int(config.get('semantic_gate_freeze_steps', 0) or 0)
        self.gate_max_target = float(getattr(model, 'semantic_gate_max_target', 0.0))
        self.gate_max_warmup_steps = int(
            getattr(model, 'semantic_gate_max_warmup_steps', 0) or 0)
        self.reg_start = float(config.get('semantic_reg_coef', 0.0))
        self.reg_final = float(config.get('semantic_reg_coef_final', self.reg_start))
        self.reg_anneal_steps = int(config.get('semantic_reg_anneal_steps', 0) or 0)
        self.frozen = self.freeze_steps > 0
        if self.frozen:
            set_gate_requires_grad(model, False)
        self._last_warn_step = -10 ** 9
        log.info(
            'Gate schedule: init=%s freeze_steps=%d gate_lr_mult=%s '
            'gate_max %.4f->%.4f over %d steps reg %.5f->%.5f over %d steps',
            getattr(model, 'semantic_token_layer_gate').detach().flatten()[:1].tolist(),
            self.freeze_steps, config.get('semantic_gate_lr_mult', 1.0),
            self.gate_max_target, self.gate_max_target, self.gate_max_warmup_steps,
            self.reg_start, self.reg_final, self.reg_anneal_steps)

    def _ramp(self, step, warmup, start, end):
        if warmup <= 0:
            return end
        if step >= warmup:
            return end
        return start + (end - start) * (step / float(warmup))

    def step(self, step):
        """Apply the schedule for this step; return scalars to log."""
        if self.frozen and step >= self.freeze_steps:
            set_gate_requires_grad(self.model, True)
            self.frozen = False
            self.log.info('Gate unfrozen at step %d', step)

        gate_max = self._ramp(step, self.gate_max_warmup_steps, 0.0, self.gate_max_target)
        self.model.set_semantic_gate_max(gate_max)

        if self.reg_anneal_steps > 0 and step >= self.reg_anneal_steps:
            reg_coef = self.reg_final
        else:
            reg_coef = self._ramp(step, self.reg_anneal_steps, self.reg_start,
                                  self.reg_final)
        return {'semantic_reg_coef_current': reg_coef,
                'semantic_gate_max_current': gate_max,
                'semantic_gate_frozen': float(self.frozen)}

    def health_check(self, step, drift, layer_std, interval=500):
        """Say loudly when the gate is stuck -- the failure this schedule exists
        to prevent.  The previous run looked healthy in the CSV because only a
        single global grad_norm was logged."""
        if self.frozen or step < self.freeze_steps + interval:
            return
        if step - self._last_warn_step < interval:
            return
        if layer_std < 1e-3 and drift < 1e-3:
            self._last_warn_step = step
            self.log.warning(
                'GATE NOT LEARNING at step %d: |w-w_init|max=%.3e, '
                'layer_std_max=%.3e. Check semantic_gate_lr_mult / init.',
                step, drift, layer_std)


class ExperimentTableWriter:
    """Write spreadsheet-friendly training data without depending on TensorBoard."""

    STEP_FIELDS = (
        'run_name', 'epoch', 'epoch_step', 'global_step', 'elapsed_seconds',
        'loss', 'lr', 'grad_norm', 'gpu_memory_gb', 'gpu_peak_memory_gb',
        'semantic_object_present_ratio', 'semantic_attribute_present_ratio',
        'semantic_relation_present_ratio', 'semantic_condition_norm_global',
        'semantic_condition_norm_object', 'semantic_condition_norm_attribute',
        'semantic_condition_norm_relation', 'semantic_adapter_norm_global',
        'semantic_adapter_norm_object', 'semantic_adapter_norm_attribute',
        'semantic_adapter_norm_relation', 'semantic_time_gate_global',
        'semantic_time_gate_object', 'semantic_time_gate_attribute',
        'semantic_time_gate_relation', 'semantic_layer_scale_global',
        'semantic_layer_scale_object', 'semantic_layer_scale_attribute',
        'semantic_layer_scale_relation', 'semantic_residual_norm',
        'semantic_token_gate', 'semantic_pair_strength',
        'semantic_role_bias_object', 'semantic_role_bias_attribute',
        'semantic_role_bias_relation',
        'semantic_layer_gate_global', 'semantic_layer_gate_object',
        'semantic_layer_gate_attribute', 'semantic_layer_gate_relation',
        'diffusion_loss', 'semantic_reg_loss', 'semantic_reg_coef',
        'distill_loss', 'distill_coef',
        # ---- TP-SCDA gate observability (see results/GATE_OPTIMIZATION.md) ----
        # Without these the previous run looked fine while the gate never moved:
        # only a single global grad_norm was recorded.
        'semantic_gate_raw', 'semantic_gate_effective',
        'semantic_gate_max_current', 'semantic_gate_frozen',
        'semantic_gate_drift_max', 'semantic_gate_layer_std_max',
        'semantic_gate_drift_global', 'semantic_gate_drift_object',
        'semantic_gate_drift_attribute', 'semantic_gate_drift_relation',
        'semantic_gate_layer_std_global', 'semantic_gate_layer_std_object',
        'semantic_gate_layer_std_attribute', 'semantic_gate_layer_std_relation',
        'semantic_gate_grad_norm', 'semantic_token_gate_grad_norm',
        'semantic_gate_grad_share', 'semantic_reg_coef_current',
        'semantic_reg_share', 'skipped_batches',
        'semantic_binding_loss', 'semantic_binding_coef',
    )

    def __init__(self, work_dir, run_name, config_dict):
        self.output_dir = Path(work_dir) / 'experiment_tables'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.step_path = self.output_dir / 'training_metrics.csv'
        self.epoch_path = self.output_dir / 'epoch_summary.csv'
        self._step_file = self.step_path.open('a', newline='', encoding='utf-8')
        self._epoch_file = self.epoch_path.open('a', newline='', encoding='utf-8')
        self._step_writer = csv.DictWriter(self._step_file, fieldnames=self.STEP_FIELDS)
        self._epoch_writer = csv.DictWriter(self._epoch_file, fieldnames=self.STEP_FIELDS)
        if self.step_path.stat().st_size == 0:
            self._step_writer.writeheader()
        if self.epoch_path.stat().st_size == 0:
            self._epoch_writer.writeheader()
        metadata_path = self.output_dir / f'{run_name}_metadata.json'
        if not metadata_path.exists():
            metadata_path.write_text(json.dumps(config_dict, indent=2, default=str), encoding='utf-8')

    def write_step(self, values):
        self._step_writer.writerow({field: values.get(field, '') for field in self.STEP_FIELDS})
        self._step_file.flush()

    def write_epoch(self, values):
        self._epoch_writer.writerow({field: values.get(field, '') for field in self.STEP_FIELDS})
        self._epoch_file.flush()

    def close(self):
        self._step_file.close()
        self._epoch_file.close()


def semantic_mask_presence(data_info):
    if not isinstance(data_info, dict) or 'semantic_token_masks' not in data_info:
        return {}
    masks = data_info['semantic_token_masks']
    if masks.ndim != 3 or masks.shape[1] != 3:
        return {}
    names = ('object', 'attribute', 'relation')
    return {
        f'semantic_{name}_present_ratio': (masks[:, index].sum(dim=1) > 0).float().mean().item()
        for index, name in enumerate(names)
    }

def train():
    if config.get('debug_nan', False):
        DebugUnderflowOverflow(model)
        logger.info('NaN debugger registered. Start to detect overflow during training.')
    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()

    start_step = start_epoch * len(train_dataloader)
    global_step = 0
    total_steps = len(train_dataloader) * config.num_epochs

    # Drives the injection-ceiling ramp, the semantic-L2 annealing and the gate
    # freeze window; see results/GATE_OPTIMIZATION.md.
    gate_schedule = None
    if getattr(accelerator.unwrap_model(model), 'semantic_token_attention_enabled', False):
        gate_schedule = GateTrainingSchedule(
            accelerator.unwrap_model(model), config, logger)

    # Optional hard stop, so a probe run does not fill the disk with 5 GB
    # checkpoints (each save carries model + EMA + optimiser state).
    max_train_steps = int(config.get('max_train_steps', 0) or 0)
    reached_step_cap = False
    skipped_batches = 0
    if max_train_steps:
        logger.info('max_train_steps=%d configured; training will stop there.',
                    max_train_steps)

    load_vae_feat = getattr(train_dataloader.dataset, 'load_vae_feat', False)
    # Now you train the model
    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        data_time_start= time.time()
        data_time_all = 0
        epoch_sums = {}
        epoch_records = 0
        for step, batch in enumerate(train_dataloader):
            data_time_all += time.time() - data_time_start
            if load_vae_feat:
                z = batch[0]
            else:
                with torch.no_grad():
                    with accelerator.autocast():
                        posterior = vae.encode(batch[0]).latent_dist
                        if config.sample_posterior:
                            z = posterior.sample()
                        else:
                            z = posterior.mode()
            clean_images = z * config.scale_factor
            y = batch[1]
            y_mask = batch[2]
            data_info = batch[3]

            # Sample a random timestep for each image
            bs = clean_images.shape[0]
            timesteps = torch.randint(0, config.train_sampling_steps, (bs,), device=clean_images.device).long()
            grad_norm = None
            schedule_values = {}
            if gate_schedule is not None:
                schedule_values = gate_schedule.step(global_step + start_step)
            with accelerator.accumulate(model):
                # Predict the noise residual
                optimizer.zero_grad()
                model_kwargs = dict(y=y, mask=y_mask, data_info=data_info)
                distill_coef = float(config.get('distill_coef', 0.0))
                loss_term = train_diffusion.training_losses(
                    model, clean_images, timesteps, model_kwargs=model_kwargs,
                    return_model_output=(teacher_model is not None and distill_coef > 0.0))
                diffusion_loss = loss_term['loss'].mean()
                semantic_reg_coef = schedule_values.get(
                    'semantic_reg_coef_current',
                    float(config.get('semantic_reg_coef', 0.0)))
                semantic_reg = accelerator.unwrap_model(model).get_semantic_regularization()
                if semantic_reg is None:
                    semantic_reg = diffusion_loss.new_zeros(())
                semantic_reg_loss = semantic_reg.to(diffusion_loss.dtype)
                # B2: explicit object->attribute alignment objective.
                semantic_binding_coef = float(config.get('semantic_binding_coef', 0.0))
                binding_raw = accelerator.unwrap_model(model).get_semantic_binding_loss()
                if semantic_binding_coef > 0.0 and binding_raw is not None:
                    semantic_binding_loss = binding_raw.to(diffusion_loss.dtype)
                else:
                    semantic_binding_loss = diffusion_loss.new_zeros(())
                # Teacher evaluation can be sampled sparsely to keep the
                # frozen-teacher refinement practical on a single GPU.
                distill_interval = max(1, int(config.get('distill_interval', 1)))
                do_distill = (teacher_model is not None and distill_coef > 0.0
                              and ((global_step + start_step) % distill_interval == 0))
                if do_distill:
                    student_output = loss_term['_model_output'][:, :clean_images.shape[1]]
                    x_t = loss_term['_x_t']
                    with torch.no_grad():
                        teacher_output = teacher_model(x_t, timesteps, **model_kwargs)
                        if isinstance(teacher_output, dict) and teacher_output.get('x', None) is not None:
                            teacher_output = teacher_output['x']
                        teacher_output = teacher_output[:, :clean_images.shape[1]]
                    distill_loss = (student_output.float() - teacher_output.float()).pow(2).mean()
                else:
                    distill_loss = diffusion_loss.new_zeros(())
                loss = (diffusion_loss + semantic_reg_coef * semantic_reg_loss
                        + semantic_binding_coef * semantic_binding_loss
                        + distill_coef * distill_loss)
                accelerator.backward(loss)
                # Gate gradient magnitudes, measured BEFORE clipping.  The old
                # logs only carried one global grad_norm, which is why a gate
                # that never moved still looked like a healthy run.
                grad_model = accelerator.unwrap_model(model)
                gate_grad_norm = parameter_grad_norm(
                    grad_model, ('semantic_token_layer_gate',))
                token_gate_grad_norm = parameter_grad_norm(
                    grad_model, ('semantic_token_gate',))
                proj_grad_norm = parameter_grad_norm(
                    grad_model, tuple(
                        name for name, _ in grad_model.named_parameters()
                        if name.startswith('semantic_token_cross_attention')))
                # Decide BEFORE clipping -- see all_grads_finite() for why.
                loss_finite = bool(torch.isfinite(loss.detach()))
                grads_finite = loss_finite and all_grads_finite(grad_model)
                if accelerator.sync_gradients and grads_finite:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)
                # Never update weights from a non-finite fp16 batch. This was
                # observed in gate_v2 and otherwise silently poisons training.
                finite = grads_finite
                if grad_norm is not None:
                    finite = finite and bool(torch.isfinite(torch.as_tensor(grad_norm)))
                if finite:
                    optimizer.step()
                    lr_scheduler.step()
                    if accelerator.sync_gradients:
                        ema_update(model_ema, model, config.ema_rate)
                else:
                    optimizer.zero_grad(set_to_none=True)
                    skipped_batches += 1
                    logger.warning(
                        'Skipping non-finite batch at epoch=%d step=%d '
                        '(loss_finite=%s grads_finite=%s; %d skipped so far)',
                        epoch, step + 1, loss_finite, grads_finite, skipped_batches)

            lr = lr_scheduler.get_last_lr()[0]
            logs = {args.loss_report_name: accelerator.gather(loss).mean().item(),
                    'diffusion_loss': accelerator.gather(diffusion_loss).mean().item(),
                    'semantic_reg_loss': accelerator.gather(semantic_reg_loss).mean().item(),
                    'semantic_reg_coef': semantic_reg_coef,
                    'semantic_binding_loss': accelerator.gather(semantic_binding_loss).mean().item(),
                    'semantic_binding_coef': semantic_binding_coef,
                    'distill_loss': accelerator.gather(distill_loss).mean().item(),
                    'distill_coef': distill_coef}
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            logs.update(schedule_values)
            logs['semantic_gate_grad_norm'] = gate_grad_norm
            logs['semantic_token_gate_grad_norm'] = token_gate_grad_norm
            # How much of the semantic branch's gradient actually reaches the
            # gates; ~0 means the gates are effectively disconnected.
            logs['semantic_gate_grad_share'] = (
                gate_grad_norm / proj_grad_norm if proj_grad_norm > 0 else 0.0)
            # How much of the total loss the semantic L2 term actually is.  The
            # coefficient had to be rescaled because opening the gates by ~10x
            # makes the residual norm ~29x larger, i.e. its square ~835x larger.
            logs['semantic_reg_share'] = (
                (semantic_reg_coef * logs['semantic_reg_loss'])
                / max(logs.get('diffusion_loss', 0.0), 1e-12))
            logs['skipped_batches'] = float(skipped_batches)
            semantic_stats = accelerator.unwrap_model(model).get_semantic_stats()
            if semantic_stats is not None:
                for name, value in semantic_stats.items():
                    logs[name] = accelerator.gather(value.reshape(1)).mean().item()
            if gate_schedule is not None and semantic_stats is not None:
                gate_schedule.health_check(
                    global_step + start_step,
                    float(semantic_stats.get('semantic_gate_drift_max', 0.0)),
                    float(semantic_stats.get('semantic_gate_layer_std_max', 0.0)))
            log_buffer.update(logs)
            if (step + 1) % config.log_interval == 0 or (step + 1) == 1:
                t = (time.time() - last_tic) / config.log_interval
                t_d = data_time_all / config.log_interval
                avg_time = (time.time() - time_start) / (global_step + 1)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - start_step - global_step - 1))))
                eta_epoch = str(datetime.timedelta(seconds=int(avg_time * (len(train_dataloader) - step - 1))))
                # avg_loss = sum(loss_buffer) / len(loss_buffer)
                log_buffer.average()
                info = f"Step/Epoch [{(epoch-1)*len(train_dataloader)+step+1}/{epoch}][{step + 1}/{len(train_dataloader)}]:total_eta: {eta}, " \
                       f"epoch_eta:{eta_epoch}, time_all:{t:.3f}, time_data:{t_d:.3f}, lr:{lr:.3e}, s:({accelerator.unwrap_model(model).h}, {accelerator.unwrap_model(model).w}), "
                info += ', '.join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                logger.info(info)
                # One compact line per log interval that says whether the gate is
                # actually being driven, so a stuck gate is visible in the log
                # itself and not only after the fact in the CSV.
                logger.info(
                    'GATE step=%d eff=%.5f ceil=%.4f frozen=%d raw=%.4f '
                    'drift_max=%.3e layer_std_max=%.3e grad=%.3e grad_share=%.3e',
                    global_step + start_step,
                    logs.get('semantic_gate_effective', float('nan')),
                    logs.get('semantic_gate_max_current', float('nan')),
                    int(logs.get('semantic_gate_frozen', 0) or 0),
                    logs.get('semantic_gate_raw', float('nan')),
                    logs.get('semantic_gate_drift_max', float('nan')),
                    logs.get('semantic_gate_layer_std_max', float('nan')),
                    logs.get('semantic_gate_grad_norm', float('nan')),
                    logs.get('semantic_gate_grad_share', float('nan')))
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
            logs.update(lr=lr)
            accelerator.log(logs, step=global_step + start_step)

            metric_values = {
                'run_name': experiment_run_name,
                'epoch': epoch,
                'epoch_step': step + 1,
                'global_step': global_step + start_step + 1,
                'elapsed_seconds': round(time.time() - time_start, 4),
                'loss': logs.get(args.loss_report_name),
                'lr': lr,
                'grad_norm': logs.get('grad_norm', ''),
                'gpu_memory_gb': round(torch.cuda.memory_allocated() / 1024 ** 3, 4) if torch.cuda.is_available() else 0.0,
                'gpu_peak_memory_gb': round(torch.cuda.max_memory_allocated() / 1024 ** 3, 4) if torch.cuda.is_available() else 0.0,
            }
            metric_values.update(semantic_mask_presence(data_info))
            # NOTE: names here must be listed explicitly -- the generic
            # `semantic_*` pass below skips anything without that prefix, which
            # silently dropped `skipped_batches` from the CSV until now.
            for name in ('diffusion_loss', 'semantic_reg_loss', 'semantic_reg_coef',
                         'distill_loss', 'distill_coef', 'skipped_batches',
                         'semantic_binding_loss', 'semantic_binding_coef'):
                if name in logs:
                    metric_values[name] = logs[name]
            for name in ExperimentTableWriter.STEP_FIELDS:
                if name.startswith('semantic_') and name in logs:
                    metric_values[name] = logs[name]
            for name, value in metric_values.items():
                if isinstance(value, (float, int)):
                    epoch_sums[name] = epoch_sums.get(name, 0.0) + value
            epoch_records += 1
            if metric_writer is not None and (
                step == 0 or (step + 1) % config.get('metrics_log_interval', config.log_interval) == 0
            ):
                metric_writer.write_step(metric_values)

            global_step += 1
            data_time_start= time.time()

            if ((epoch - 1) * len(train_dataloader) + step + 1) % config.save_model_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    os.umask(0o000)
                    save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                    epoch=epoch,
                                    step=(epoch - 1) * len(train_dataloader) + step + 1,
                                    model=accelerator.unwrap_model(model),
                                    model_ema=accelerator.unwrap_model(model_ema),
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler
                                    )

            if max_train_steps and (global_step + start_step) >= max_train_steps:
                logger.info('Hit max_train_steps=%d at global_step=%d; stopping training.',
                            max_train_steps, global_step + start_step)
                reached_step_cap = True
                break

        if reached_step_cap:
            break

        if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                os.umask(0o000)
                save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                epoch=epoch,
                                step=(epoch - 1) * len(train_dataloader) + step + 1,
                                model=accelerator.unwrap_model(model),
                                model_ema=accelerator.unwrap_model(model_ema),
                                optimizer=optimizer,
                                    lr_scheduler=lr_scheduler
                                    )
        if metric_writer is not None and epoch_records:
            epoch_values = {
                'run_name': experiment_run_name,
                'epoch': epoch,
                'epoch_step': len(train_dataloader),
                'global_step': global_step + start_step,
                'elapsed_seconds': round(time.time() - time_start, 4),
            }
            epoch_values.update({name: total / epoch_records for name, total in epoch_sums.items()})
            metric_writer.write_epoch(epoch_values)


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("config", type=str, help="config")
    parser.add_argument("--cloud", action='store_true', default=False, help="cloud or local machine")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the dir to resume the training')
    parser.add_argument('--load-from', default=None, help='the dir to load a ckpt for training')
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--loss_report_name", type=str, default="loss")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config)
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        config.work_dir = args.work_dir
    if args.cloud:
        config.data_root = '/data/data'
    if args.resume_from is not None:
        config.load_from = None
        config.resume_from = dict(
            checkpoint=args.resume_from,
            load_ema=False,
            resume_optimizer=True,
            resume_lr_scheduler=True)
    if args.debug:
        config.log_interval = 1
        config.train_batch_size = 8
        config.valid_num = 100

    if not torch.cuda.is_available():
        if config.use_fsdp:
            raise ValueError('FSDP requires a CUDA device; set use_fsdp=False for CPU training.')
        if config.mixed_precision != 'no':
            config.mixed_precision = 'no'

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug
    # Initialize accelerator and tensorboard logging
    if config.use_fsdp:
        init_train = 'FSDP'
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),)
    else:
        init_train = 'DDP'
        fsdp_plugin = None

    even_batches = True
    if config.multi_scale:
        even_batches=False,

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=os.path.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        even_batches=even_batches,
        kwargs_handlers=[init_handler]
    )

    logger = get_root_logger(os.path.join(config.work_dir, 'train_log.log'))

    config.seed = init_random_seed(config.get('seed', None))
    set_random_seed(config.seed)

    if accelerator.is_main_process:
        config.dump(os.path.join(config.work_dir, 'config.py'))

    logger.info(f"Config: \n{config.pretty_text}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.seed}")
    logger.info(f"Initializing: {init_train} for training")
    image_size = config.image_size  # @param [256, 512, 1024]
    latent_size = int(image_size) // 8
    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
    model_kwargs={"window_block_indexes": config.window_block_indexes, "window_size": config.window_size,
                  "use_rel_pos": config.use_rel_pos, "lewei_scale": config.lewei_scale, 'config':config,
                  'model_max_length': config.model_max_length,
                  'semantic_conditioning': config.get('semantic_conditioning', False),
                  'semantic_adapter_dim': config.get('semantic_adapter_dim', 64),
                  'semantic_dropout': config.get('semantic_dropout', 0.1),
                  'time_condition_gate': config.get('time_condition_gate', True),
                  'condition_gate_scale': config.get('condition_gate_scale', 0.05)}
    model_kwargs.update({'semantic_residual_scale': config.get('semantic_residual_scale', 1.0),
                         'semantic_active_branches': config.get('semantic_active_branches', None),
                         'semantic_token_attention': config.get('semantic_token_attention', False),
                         'semantic_token_gate_max': config.get('semantic_token_gate_max', 1.0),
                         # Gate controls; defaults reproduce the model's own
                         # defaults, so omitting them from a config is safe.
                         'semantic_gate_init': config.get('semantic_gate_init', -4.0),
                         'semantic_gate_init_global': config.get('semantic_gate_init_global', 4.0),
                         'semantic_token_gate_init': config.get('semantic_token_gate_init', 0.05),
                         'semantic_token_gate_activation': config.get('semantic_token_gate_activation', 'tanh'),
                         'semantic_gate_max_warmup_steps': config.get('semantic_gate_max_warmup_steps', 0),
                         # B1 / B2
                         'semantic_pair_edge_gate': config.get('semantic_pair_edge_gate', False),
                         'cross_attn_pair_replace': config.get('cross_attn_pair_replace', False),
                         'semantic_binding_coef': config.get('semantic_binding_coef', 0.0),
                         'semantic_binding_temperature': config.get('semantic_binding_temperature', 1.0)})
    logger.info('Gate configuration reaching the model: init=%s/%s raw_init=%s '
                'activation=%s ceiling=%s warmup=%s',
                model_kwargs['semantic_gate_init'], model_kwargs['semantic_gate_init_global'],
                model_kwargs['semantic_token_gate_init'],
                model_kwargs['semantic_token_gate_activation'],
                model_kwargs['semantic_token_gate_max'],
                model_kwargs['semantic_gate_max_warmup_steps'])

    # build models
    train_diffusion = IDDPM(str(config.train_sampling_steps), learn_sigma=learn_sigma, pred_sigma=pred_sigma, snr=config.snr_loss)
    model = build_model(config.model,
                        config.grad_checkpointing,
                        config.get('fp32_attention', False),
                        input_size=latent_size,
                        learn_sigma=learn_sigma,
                        pred_sigma=pred_sigma,
                        **model_kwargs).train()
    logger.info(f"{model.__class__.__name__} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    model_ema = deepcopy(model).eval()

    if config.load_from is not None:
        if args.load_from is not None:
            config.load_from = args.load_from
        missing, unexpected = load_checkpoint(config.load_from, model, load_ema=config.get('load_ema', False))
        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')

    configure_trainable_parameters(model, config.get('train_semantic_only', False),
                                   config.get('trainable_contains'))

    # Build a frozen baseline teacher after loading the student checkpoint.
    # The teacher uses the same PixArt weights but has all semantic paths
    # disabled, so distillation preserves the original model's behavior.
    if config.get('distill_teacher_checkpoint', None) and config.get('distill_coef', 0.0) > 0:
        teacher_kwargs = dict(model_kwargs)
        teacher_kwargs.update({
            'semantic_conditioning': False,
            'semantic_token_attention': False,
            'semantic_residual_scale': 0.0,
        })
        teacher_model = build_model(config.model,
                                    config.grad_checkpointing,
                                    config.get('fp32_attention', False),
                                    input_size=latent_size,
                                    learn_sigma=learn_sigma,
                                    pred_sigma=pred_sigma,
                                    **teacher_kwargs).to(accelerator.device).eval()
        teacher_missing, teacher_unexpected = load_checkpoint(
            config.distill_teacher_checkpoint, teacher_model, load_ema=False)
        teacher_model.requires_grad_(False)
        logger.info('Loaded frozen distillation teacher from %s (missing=%d unexpected=%d)',
                    config.distill_teacher_checkpoint, len(teacher_missing), len(teacher_unexpected))

    ema_update(model_ema, model, 0.)
    if not config.data.load_vae_feat:
        vae = AutoencoderKL.from_pretrained(config.vae_pretrained).to(accelerator.device).eval()

    # prepare for FSDP clip grad norm calculation
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)

    # build dataloader
    set_data_root(config.data_root)
    dataset = build_dataset(config.data, resolution=image_size, aspect_ratio_type=config.aspect_ratio_type)
    if config.multi_scale:
        batch_sampler = AspectRatioBatchSampler(sampler=RandomSampler(dataset), dataset=dataset,
                                                batch_size=config.train_batch_size, aspect_ratios=dataset.aspect_ratio, drop_last=True,
                                                ratio_nums=dataset.ratio_nums, config=config, valid_num=config.valid_num)
        # used for balanced sampling
        # batch_sampler = BalancedAspectRatioBatchSampler(sampler=RandomSampler(dataset), dataset=dataset,
        #                                                 batch_size=config.train_batch_size, aspect_ratios=dataset.aspect_ratio,
        #                                                 ratio_nums=dataset.ratio_nums)
        train_dataloader = build_dataloader(
            dataset, batch_sampler=batch_sampler, num_workers=config.num_workers,
            persistent_workers=config.get('persistent_workers', config.num_workers > 0),
            prefetch_factor=config.get('prefetch_factor', None))
    else:
        train_dataloader = build_dataloader(
            dataset, num_workers=config.num_workers, batch_size=config.train_batch_size, shuffle=True,
            persistent_workers=config.get('persistent_workers', config.num_workers > 0),
            prefetch_factor=config.get('prefetch_factor', None))

    # build optimizer and lr scheduler
    lr_scale_ratio = 1
    if config.get('auto_lr', None):
        lr_scale_ratio = auto_scale_lr(config.train_batch_size * get_world_size() * config.gradient_accumulation_steps,
                                       config.optimizer, **config.auto_lr)
    optimizer = build_optimizer(model, config.optimizer)
    # --- gate-specific learning rate -----------------------------------------
    # auto_scale_lr() has already rewritten config.optimizer['lr'] in place, so
    # the value read here is the effective one.  The 113 gate scalars get their
    # own group; sharing 3.5e-6 with the 8M-parameter projections is what kept
    # them pinned at their initialisation (results/GATE_OPTIMIZATION.md).
    gate_lr_mult = float(config.get('semantic_gate_lr_mult', 1.0))
    if gate_lr_mult != 1.0:
        gate_params, other_params = split_gate_parameters(model)
        if gate_params:
            base_lr = float(config.optimizer.get('lr', 1e-4))
            optimizer = torch.optim.AdamW(
                [dict(params=other_params, lr=base_lr),
                 dict(params=gate_params, lr=base_lr * gate_lr_mult)],
                lr=base_lr,
                weight_decay=config.optimizer.get('weight_decay', 0.0),
                eps=config.optimizer.get('eps', 1e-8))
            logger.info(
                'Gate lr group: %d tensors at lr=%.3e (x%.1f); %d other tensors at %.3e',
                len(gate_params), base_lr * gate_lr_mult, gate_lr_mult,
                len(other_params), base_lr)
        else:
            logger.warning('semantic_gate_lr_mult set but no trainable gate '
                           'parameters found (are they frozen?)')
    lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio)

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")

    experiment_run_name = config.get('experiment_name', Path(args.config).stem)
    metric_writer = None
    if accelerator.is_main_process and config.get('save_experiment_tables', True):
        metric_writer = ExperimentTableWriter(config.work_dir, experiment_run_name, dict(vars(config)))
        logger.info(f'Experiment tables: {metric_writer.output_dir}')

    start_epoch = 0
    if config.resume_from is not None and config.resume_from['checkpoint'] is not None:
        start_epoch, missing, unexpected = load_checkpoint(**config.resume_from,
                                                           model=model,
                                                           model_ema=model_ema,
                                                           optimizer=optimizer,
                                                           lr_scheduler=lr_scheduler,
                                                           )

        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, model_ema = accelerator.prepare(model, model_ema)
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)
    try:
        train()
    finally:
        if metric_writer is not None:
            metric_writer.close()

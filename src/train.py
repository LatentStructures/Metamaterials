"""Training loop for the baseline DDPM (ROADMAP section 4.5, section 5).

Reproducible from train_baseline.yaml + model_baseline.yaml + a fixed seed.
Supports bf16 autocast, EMA, checkpoint/resume, cosine LR, wandb/tensorboard,
periodic DDIM sampling with rendered voxel-grid images, and an ``--overfit``
mode for the 10-sample architecture test (section 4.2). No hyperparameters are
hardcoded here.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.config import REPO_ROOT, check_conditioning_order, load_yaml, set_seed
from src.data.dataset import MetamaterialDataset, VoxelTransform
from src.models.diffusion import build_diffusion
from src.models.unet3d import UNet3D

LOG = logging.getLogger(__name__)

# Default power-iteration budget for the Option-B connectivity proxy inside the
# autograd graph. 60 steps converges the proxy eigenvalue tightly enough for the
# regime loss at a fraction of the 200-step VRAM/deep-graph cost; override with
# ``physics_loss.iterations`` in the train config (perf block).
DEFAULT_PHYSICS_ITERATIONS = 60


class EMA:
    """Exponential moving average of the model weights.

    NOTE: at ``decay=0.9999`` over a 30k-step cosine-annealed run the shadow is a
    10k-step average centred ~step 25k -- before the LR reaches 0 -- so it lags the
    fully-annealed live weights into a less-converged region and generates
    fragmented grids (~14% valid vs 92% live). Sample with ``--weights live``.
    See ``PROJECT.md`` §8.
    """

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply(self, model: torch.nn.Module) -> None:
        model.load_state_dict({k: v.float() for k, v in self.shadow.items()}, strict=True)

    @contextmanager
    def average_parameters(self, model: torch.nn.Module):
        """Temporarily apply EMA parameters and restore live weights on exit."""
        raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.apply(model)
        try:
            yield
        finally:
            model.load_state_dict(raw, strict=True)


def _rng_state() -> dict:
    """RNG state in tensors/primitives so checkpoints remain weights-only safe."""
    py_version, py_state, py_gauss = random.getstate()
    np_name, np_keys, np_pos, np_has_gauss, np_cached = np.random.get_state()
    state = {
        "torch_cpu": torch.get_rng_state(),
        "python": [py_version, list(py_state), py_gauss],
        "numpy": [np_name, torch.from_numpy(np_keys.copy()), int(np_pos),
                  int(np_has_gauss), float(np_cached)],
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch_cpu"].cpu())
    py_version, py_state, py_gauss = state["python"]
    random.setstate((int(py_version), tuple(int(v) for v in py_state), py_gauss))
    np_name, np_keys, np_pos, np_has_gauss, np_cached = state["numpy"]
    np.random.set_state((np_name, np_keys.cpu().numpy().astype(np.uint32),
                         int(np_pos), int(np_has_gauss), float(np_cached)))
    if torch.cuda.is_available() and "torch_cuda" in state:
        # CUDA accepts only CPU ByteTensors here; a checkpoint loaded with
        # map_location=device would otherwise hand it GPU tensors.
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])


def render_voxel_projection(vox: torch.Tensor) -> np.ndarray:
    """Orthographic max-intensity projections -> (3, 32, 32) CHW uint8 image."""
    v = vox.detach().float().cpu()
    if v.ndim == 5:
        v = v[0, 0]
    elif v.ndim == 4:
        v = v[0]
    v = (v > 0.5).float()
    xs = [np.asarray(v.max(dim=k).values) for k in (0, 1, 2)]
    return (np.stack(xs, axis=0) * 255.0).astype(np.uint8)


def _make_logger(backend: str, project: str, run_dir: Path, config: dict):
    """Return (log_scalar, log_image) with wandb->tensorboard fallback."""
    if backend == "wandb":
        try:
            import wandb
            tags = ([config.get("physics_ablation_tag", "abl_physics")]
                    if config.get("physics_ablation") else None)
            wandb.init(project=project, config=config, dir=str(run_dir.parent), tags=tags)
            return (lambda tag, val, step: wandb.log({tag: val}, step=step),
                    lambda tag, img, step: wandb.log({tag: wandb.Image(img)}, step=step))
        except Exception as e:  # degraded-but-usable fallback
            LOG.warning("wandb unavailable (%s); falling back to tensorboard", e)
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(log_dir=str(run_dir))
    tb.add_text("run/config", str(config))
    return (tb.add_scalar, tb.add_image)


def _load_physics(lamb: float, device: torch.device, iterations: int):
    """Option B connectivity proxy (ROADMAP 4.4); disabled when lambda_1 == 0.

    ``iterations`` bounds the deflated power iteration inside the autograd
    graph: more iterations converge tighter but keep a deeper graph in VRAM.
    """
    if lamb <= 0.0:
        return lambda *_: torch.tensor(0.0, device=device)
    try:
        from src.losses.physics_loss import connectivity_proxy_loss
    except ImportError:
        raise SystemExit(
            f"physics_loss.lambda_1 = {lamb} > 0 requires "
            "src/losses/physics_loss.py (Option B, ROADMAP section 4.4) "
            "and cannot run without it."
        ) from None
    return lambda x, c, t: connectivity_proxy_loss(x, c, t, iterations=int(iterations))


def _resolve_physics_lambda(physics_cfg: dict, ablation: bool = False) -> float:
    """Choose the Option-B baseline weight or its configured zero-loss control.

    Also enforces ``physics_loss.option``: only Option B is implemented, so an
    ``option: A`` config must fail loudly instead of silently running B.
    """
    option = physics_cfg.get("option", "B")
    if option != "B":
        raise ValueError(
            f"physics_loss.option = {option!r} is not supported; only Option B "
            "is implemented (ROADMAP section 4.4, Option A is a contingency)"
        )
    if not ablation:
        return float(physics_cfg.get("lambda_1", 0.0))
    ablation_cfg = physics_cfg.get("ablation", {})
    if "lambda_1" not in ablation_cfg:
        raise ValueError("--physics-ablation requires physics_loss.ablation.lambda_1")
    return float(ablation_cfg["lambda_1"])


def _make_augment(enabled: bool, seed: int) -> VoxelTransform | None:
    """Deterministic proper-rotation augment keyed by the sample index.

    Each sample is rotated by one of the 24 cube rotations (never a mirror),
    derived from ``seed + sample_index``. Making the rotation a pure function
    of the sample index keeps the augmentation reproducible run-to-run even
    with ``num_workers > 0`` (every worker derives the same rotation for the
    same index, instead of sharing one mutable RNG sequence whose interleaving
    depends on worker scheduling). The conditioning vector [E, rho, nu] is
    invariant under proper rotations for the baseline families, so the label
    stays valid.
    """
    if not enabled:
        return None
    from src.data.augment import random_rotation, rotate_voxels

    def transform(v: np.ndarray, idx: int) -> np.ndarray:
        rng = np.random.default_rng(seed + idx)
        return rotate_voxels(v, random_rotation(rng))

    return transform


def _clip_grad(model: torch.nn.Module, max_norm: float | None) -> float | None:
    """Config-driven global gradient norm clipping (``optimizer.grad_clip``).

    Returns the pre-clip total gradient norm, or None when clipping is
    disabled (``0``/absent) so the logging path stays branch-free.
    """
    if not max_norm or max_norm <= 0.0:
        return None
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))


def _parse():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-config", default=str(REPO_ROOT / "configs/train_baseline.yaml"))
    ap.add_argument("--model-config", default=str(REPO_ROOT / "configs/model_baseline.yaml"))
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--overfit", type=int, default=0,
                    help="architecture test: train on N samples only (section 4.2)")
    ap.add_argument("--max-iterations", type=int, default=0, help="0 = run to completion")
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--physics-ablation", action="store_true",
                    help="run the configured lambda_1=0 Option-B ablation")
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse()

    tcfg = load_yaml(args.train_config)
    mcfg = load_yaml(args.model_config)
    check_conditioning_order(tcfg)
    if mcfg.get("sampling", {}).get("schedule", "ddim") != "ddim":
        raise ValueError("baseline sampling.schedule must be 'ddim'")
    set_seed(int(tcfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        perf = tcfg.get("perf", {})
        torch.backends.cudnn.benchmark = bool(perf.get("cudnn_benchmark", True))  # fixed 32³ shapes amortize the autotune
        torch.backends.cuda.matmul.allow_tf32 = bool(perf.get("tf32", True))
        torch.set_float32_matmul_precision(perf.get("float32_matmul_precision", "high"))  # TF32 tensor cores on Ampere
        torch.backends.cudnn.allow_tf32 = bool(perf.get("tf32", True))
    mp = tcfg.get("mixed_precision", "bf16")

    dataset = MetamaterialDataset(
        REPO_ROOT / tcfg["data"]["hdf5"],
        transform=_make_augment(tcfg["data"].get("augmentation", False),
                                int(tcfg["seed"])),
    )
    min_samples = int(tcfg["data"].get("min_samples", 0))
    if not args.overfit and len(dataset) < min_samples:
        raise RuntimeError(
            f"training split has {len(dataset)} cells; config requires at least "
            f"{min_samples}. Regenerate the dataset before production training."
        )
    if args.overfit:
        dataset.voxels = dataset.voxels[:args.overfit]
        dataset.cond = dataset.cond[:args.overfit]
        bs = min(int(tcfg["batch_size"]), args.overfit)
    else:
        bs = int(tcfg["batch_size"])
    perf = tcfg.get("perf", {})
    nw = int(perf.get("num_workers", 0))
    loader_rng = torch.Generator()
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=False,
                        num_workers=nw,
                        prefetch_factor=int(perf["prefetch_factor"]) if nw > 0 else None,
                        persistent_workers=bool(perf.get("persistent_workers", False)) if nw > 0 else False,
                        pin_memory=device.type == "cuda", generator=loader_rng)

    model = UNet3D.from_config(mcfg["unet3d"]).to(device)
    diffusion = build_diffusion(mcfg["diffusion"], device)
    n_params = sum(p.numel() for p in model.parameters())
    LOG.info("device=%s param_count=%.2fM samples=%d", device, n_params / 1e6, len(dataset))

    iters_per_epoch = max(1, len(loader))
    if args.max_iterations:
        max_iters = args.max_iterations
    elif args.overfit:
        max_iters = args.overfit * 50
    else:
        max_iters = iters_per_epoch * int(tcfg.get("epochs", 1000))
    optimizer_cfg = tcfg["optimizer"]
    opt = optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg["lr"]),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.01)),
        betas=tuple(float(v) for v in optimizer_cfg.get("betas", (0.9, 0.999))),
        eps=float(optimizer_cfg.get("eps", 1e-8)),
    )
    grad_clip = float(optimizer_cfg.get("grad_clip", 0.0) or 0.0)  # 0/absent = off
    if tcfg["optimizer"].get("lr_schedule") == "cosine":
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, max_iters))
    else:
        sched = None

    ema_cfg = tcfg.get("ema", {})
    ema = EMA(model, float(ema_cfg["decay"])) if ema_cfg.get("enabled", False) else None

    physics_cfg = tcfg.get("physics_loss", {})
    lamb = _resolve_physics_lambda(physics_cfg, ablation=args.physics_ablation)
    if lamb > 0.0 and grad_clip <= 0.0:
        # Footgun guard: the Option-B gradient has unbounded stochastic bursts
        # (pre-clip norms in the thousands) and diverged the first full run
        # without clipping; a config that omits grad_clip silently gets 0.0.
        LOG.warning(
            "physics_loss.lambda_1 = %g > 0 with optimizer.grad_clip = %g: "
            "the first full run diverged under these settings; set "
            "optimizer.grad_clip: 1.0 (see configs/train_baseline.yaml)",
            lamb, grad_clip,
        )
    physics = _load_physics(lamb, device,
                            int(physics_cfg.get("iterations",
                                                DEFAULT_PHYSICS_ITERATIONS)))
    warmup_iters = int(float(physics_cfg.get("warmup_fraction", 0.0)) * max_iters)

    run_dir = Path(args.run_dir) if args.run_dir else REPO_ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {"train": tcfg, "model": mcfg,
                  "physics_lambda_1": lamb,
                  "physics_ablation": bool(args.physics_ablation),
                  "physics_ablation_tag": physics_cfg.get("ablation", {}).get(
                      "tag", "abl_physics")}
    # Snapshot the resolved config into the run dir. Without this, reproducing a
    # run depends on the working tree at the time, which drifts: the divergence
    # fix (grad_clip) landed after the first full run and is not recoverable from
    # that run's checkpoint alone.
    with (run_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(run_config, fh, indent=2, default=str)
    log_scalar, log_image = _make_logger(
        tcfg["logging"]["backend"], tcfg["logging"]["project"], run_dir, run_config)
    ckpt_every = int(tcfg.get("checkpoint_every", 1000))
    sample_every = int(tcfg["logging"].get("sample_interval", 5000))

    autocast = torch.autocast(device_type=device.type,
                              dtype=torch.bfloat16) if mp == "bf16" else torch.autocast(
        device_type=device.type, enabled=False)

    global_step = 0
    if args.resume_from:
        ck = torch.load(args.resume_from, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        if ema:
            ema_shadow = ck.get("ema")
            if ema_shadow is not None:
                ema.shadow = {k: v.detach().clone() for k, v in ema_shadow.items()}
        opt.load_state_dict(ck["optim"])
        if sched is not None and ck.get("scheduler") is not None:
            sched.load_state_dict(ck["scheduler"])
        global_step = int(ck.get("iter", 0))
        if ck.get("rng") is not None:
            _restore_rng_state(ck["rng"])
        LOG.info("resumed from %s (iter %d)", args.resume_from, global_step)

    t_start = time.time()
    model.train()
    n_eval = min(4, len(dataset))
    cond_test = torch.as_tensor(dataset.cond[:n_eval].copy()).to(device)
    while global_step < max_iters:
        epoch = global_step // iters_per_epoch
        batch_offset = global_step % iters_per_epoch
        loader_rng.manual_seed(int(tcfg["seed"]) + epoch)
        for batch_idx, (x, c) in enumerate(loader):
            if batch_idx < batch_offset:
                continue
            if global_step >= max_iters:
                break
            x, c = x.to(device), c.to(device)
            opt.zero_grad()
            with autocast:
                # Inline DDPM step so the physics term can be attached to the
                # denoised x0 prediction (ROADMAP 4.4: proxy sees the soft,
                # pre-threshold occupancy), not to the noise-free data sample.
                b = x.shape[0]
                tt = torch.randint(0, diffusion.T, (b,), device=device)
                noise = torch.randn_like(x)
                xt = diffusion.q_sample(x, tt, noise)
                pred = model(xt, tt, c)
                loss = F.mse_loss(pred, noise)
                if lamb > 0.0:
                    lam = lamb * min(1.0, global_step / max(1, warmup_iters))
                    x0 = diffusion.predict_denoised(xt, tt, pred).clamp(0.0, 1.0)
                    loss = loss + lam * physics(x0, c, tt.to(c.dtype).unsqueeze(-1))
            loss.backward()
            grad_norm = _clip_grad(model, grad_clip)
            opt.step()
            if sched is not None:
                sched.step()
            if ema is not None:
                ema.update(model)
            global_step += 1

            if global_step % 50 == 0 or global_step == 1:
                lr = opt.param_groups[0]["lr"]
                gc_note = f" grad_preclip {grad_norm:.3e}" if grad_norm is not None else ""
                LOG.info("iter %d/%d loss %.4f lr %.2e%s elapsed %.1fs",
                         global_step, max_iters, loss.item(), lr, gc_note,
                         time.time() - t_start)
                log_scalar("train/loss", loss.item(), global_step)
                if grad_norm is not None:
                    log_scalar("train/grad_norm_preclip", grad_norm, global_step)

            if global_step % ckpt_every == 0:
                torch.save({"model": model.state_dict(), "optim": opt.state_dict(),
                            "scheduler": sched.state_dict() if sched else None,
                            "ema": dict(ema.shadow) if ema else None,
                            "rng": _rng_state(), "iter": global_step}, run_dir / "ckpt.pt")

            if global_step % sample_every == 0 or global_step == max_iters:
                model.eval()
                sampling = mcfg.get("sampling", {})
                steps = max(50, min(int(sampling.get("steps", 50)), 100))  # DDIM 50-100 locked
                eval_seed = int(sampling.get("eval_seed", 0))
                with torch.no_grad():
                    if ema is not None:
                        with ema.average_parameters(model):
                            gen = diffusion.sample_ddim(model, cond_test,
                                                        (n_eval, 1, 32, 32, 32),
                                                        steps=steps, eta=float(sampling.get("eta", 0.0)),
                                                        seed=eval_seed)
                    else:
                        gen = diffusion.sample_ddim(model, cond_test,
                                                    (n_eval, 1, 32, 32, 32),
                                                    steps=steps, eta=float(sampling.get("eta", 0.0)),
                                                    seed=eval_seed)
                img = render_voxel_projection(gen)
                log_image("eval/samples", img, global_step)
            model.train()

    # Always leave a resumable/final artifact, including short runs whose
    # final iteration is not aligned with checkpoint_every.
    torch.save({"model": model.state_dict(), "optim": opt.state_dict(),
                "scheduler": sched.state_dict() if sched else None,
                "ema": dict(ema.shadow) if ema else None,
                "rng": _rng_state(), "iter": global_step}, run_dir / "ckpt.pt")

    LOG.info("training done at iter %d in %.1fs", global_step, time.time() - t_start)


if __name__ == "__main__":
    main()

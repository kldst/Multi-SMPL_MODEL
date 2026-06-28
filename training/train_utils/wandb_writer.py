# Weights & Biases logger with the same interface as TensorBoardLogger
# (log / log_visuals / flush / close), so it is a drop-in replacement in the trainer.
#
# wandb requires the `step` passed to wandb.log() to be monotonically increasing for the
# whole run, but the trainer logs train and val with SEPARATE step counters (they overlap).
# To avoid dropped logs we DO NOT pass step= to wandb; instead we buffer all scalars that
# share the same incoming step into one wandb.log() call (flushed when the step changes),
# let wandb auto-increment its internal step, and record the real step as a "step" field
# used as the chart x-axis via define_metric.

import atexit
import logging
import os
from typing import Any, Dict, Optional, Union

from .distributed import get_machine_local_and_dist_rank


class WandbLogger:
    def __init__(
        self,
        path: str,
        project: str = "vggt_smplx",
        name: Optional[str] = None,
        entity: Optional[str] = None,
        mode: str = "online",
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        _, self._rank = get_machine_local_and_dist_rank()
        self._path = path
        self._run = None
        self._wandb = None
        self._buf: Dict[str, Any] = {}
        self._buf_step: Optional[int] = None

        if self._rank == 0:
            import wandb
            self._wandb = wandb
            os.makedirs(path, exist_ok=True)
            # env WANDB_MODE (if set) takes precedence over the config value.
            run_mode = os.environ.get("WANDB_MODE", mode)
            self._run = wandb.init(
                project=project, name=name, entity=entity, mode=run_mode,
                dir=path, config=dict(config) if config else None, **kwargs,
            )
            # all metrics use the real training step as x-axis
            wandb.define_metric("step")
            wandb.define_metric("*", step_metric="step")
            logging.info(f"[WandbLogger] run '{self._run.name}' (mode={run_mode}) dir={path}")
        atexit.register(self.close)

    # ---- helpers ----
    @staticmethod
    def _to_scalar(v):
        if hasattr(v, "item"):
            try:
                return v.item()
            except Exception:
                return v
        return v

    def _flush_buf(self) -> None:
        if self._run and self._buf:
            self._wandb.log(self._buf)
            self._buf = {}

    # ---- TensorBoardLogger-compatible API ----
    def log(self, name: str, data: Any, step: int) -> None:
        if not self._run:
            return
        if self._buf_step is not None and step != self._buf_step:
            self._flush_buf()
        self._buf_step = step
        self._buf[name] = self._to_scalar(data)
        self._buf["step"] = int(step)

    def log_dict(self, payload: Dict[str, Any], step: int) -> None:
        for k, v in payload.items():
            self.log(k, v, step)

    def log_visuals(self, name: str, data: Union[Any], step: int, fps: int = 4) -> None:
        if not self._run:
            return
        self._flush_buf()
        import numpy as np
        arr = data.detach().cpu().numpy() if hasattr(data, "detach") else np.asarray(data)
        ndim = arr.ndim
        if ndim == 3:                      # (C,H,W) -> (H,W,C)
            img = np.transpose(arr, (1, 2, 0)) if arr.shape[0] in (1, 3) else arr
            self._wandb.log({name: self._wandb.Image(img), "step": int(step)})
        elif ndim == 5:                    # (N,T,C,H,W) video
            vid = arr
            if vid.dtype != np.uint8:
                vid = (np.clip(vid, 0, 1) * 255).astype("uint8")
            self._wandb.log({name: self._wandb.Video(vid, fps=fps), "step": int(step)})
        else:
            logging.warning(f"[WandbLogger] unsupported visual ndim={ndim} for {name}")

    def flush(self) -> None:
        self._flush_buf()

    def close(self) -> None:
        if self._run:
            self._flush_buf()
            self._run.finish()
            self._run = None

    @property
    def writer(self):
        return self._run

    @property
    def path(self) -> str:
        return self._path

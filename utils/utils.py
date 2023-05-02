from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import traceback

import numpy as np
import torch
from loguru import logger


def checkpoint(file, model, optimizer, step):
    print("Checkpointing Model @ step %d ..." % step)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        file,
    )


def loadFromCheckpoint(file, model, optimizer, cfg):
    checkpoint = torch.load(file, map_location=torch.device(cfg.devices))
    model.load_state_dict(checkpoint["model_state_dict"])
    if not optimizer is None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epochs = checkpoint["step"]
    logger.info("Loading Model @ step %d" % (start_epochs))
    return start_epochs


def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def drawBottomBar(status):
    def print_there(x, y, text):
        sys.stdout.write("\x1b7\x1b[%d;%df%s\x1b8" % (x, y, text))
        sys.stdout.flush()

    def move(y, x):
        print("\033[%d;%dH" % (y, x))

    columns, rows = os.get_terminal_size()

    # status += "\x1B[K\n"
    status += " " * ((columns - (len(status) % columns)) % columns)
    # status += " " * (columns)

    lines = int(len(status) / columns)
    print("\n" * (lines), end="")
    print_there(rows - lines, 0, " " * columns)
    print_there(rows - lines + 1, 0,
                "\33[38;5;72m\33[48;5;234m%s\33[0m" % status)
    move(rows - lines - 1, 0)


class TrainingStatus:
    def __init__(self, num_steps, eta_interval=25, statusbar=""):

        self.eta_interval = eta_interval
        self.num_steps = num_steps

        self.etaCount = 0
        self.etaStart = time.time()
        self.duration = 0

        self.statusbar = " ".join(sys.argv)

    def tic(self):
        self.start = time.time()

    def toc(self, iter, loss):
        self.end = time.time()

        self.etaCount += 1
        if self.etaCount % self.eta_interval == 0:
            self.duration = time.time() - self.etaStart
            self.etaStart = time.time()

        etaTime = float(
            self.num_steps - iter) / self.eta_interval * self.duration
        m, s = divmod(etaTime, 60)
        h, m = divmod(m, 60)
        etaString = "%d:%02d:%02d" % (h, m, s)
        msg = "%.2f%% (%d/%d): %.3e  t %.3f  @ %s (%s)" % (
            iter * 100.0 / self.num_steps,
            iter,
            self.num_steps,
            loss,
            self.end - self.start,
            time.strftime("%a %d %H:%M:%S",
                          time.localtime(time.time() + etaTime)),
            etaString,
        )

        if "CUDA_VISIBLE_DEVICES" in os.environ:
            barText = "Command: CUDA_VISIBLE_DEVICES=%s python %s" % (
                os.environ["CUDA_VISIBLE_DEVICES"],
                self.statusbar,
            )
        else:
            barText = "Command: python %s" % (self.statusbar)
        try:
            drawBottomBar(barText)
        except:
            pass  # skip bottombar if it no output
        return msg


def colorize_np(x, cmap_name="jet", append_cbar=False):
    tick_cnt = 6
    vmin = x.min()
    vmax = x.max() + TINY_NUMBER
    if cmap_name == "Set1":
        tick_cnt = 10

    x = (x - vmin) / (vmax - vmin)
    # x = np.clip(x, 0., 1.)

    cmap = cm.get_cmap(cmap_name)
    x_new = cmap(x)[:, :, :3]

    cbar = get_vertical_colorbar(
        h=x.shape[0], vmin=vmin, vmax=vmax, cmap_name=cmap_name,
        tick_cnt=tick_cnt
    )

    if append_cbar:
        x_new = np.concatenate((x_new, np.zeros_like(x_new[:, :5, :]), cbar),
                               axis=1)
        return x_new
    else:
        return x_new, cbar


def colorize(x, cmap_name="jet", append_cbar=False):
    x = x.numpy()
    x, cbar = colorize_np(x, cmap_name, mask)
    if append_cbar:
        x = np.concatenate((x, np.zeros_like(x[:, :5, :]), cbar), axis=1)

    x = torch.from_numpy(x)
    return x


def colored_hook(home_dir):
    """Colorizes python's error message.

    Args:
      home_dir: directory where code resides (to highlight your own files).
    Returns:
      The traceback hook.
    """

    def hook(type_, value, tb):
        def colorize(text, color, own=0):
            """Returns colorized text."""
            endcolor = "\x1b[0m"
            codes = {
                "green": "\x1b[0;32m",
                "green_own": "\x1b[1;32;40m",
                "red": "\x1b[0;31m",
                "red_own": "\x1b[1;31m",
                "yellow": "\x1b[0;33m",
                "yellow_own": "\x1b[1;33m",
                "black": "\x1b[0;90m",
                "black_own": "\x1b[1;90m",
                "cyan": "\033[1;36m",
            }
            return codes[color + ("_own" if own else "")] + text + endcolor

        for filename, line_num, func, text in traceback.extract_tb(tb):
            basename = os.path.basename(filename)
            own = (home_dir in filename) or ("/" not in filename)

            print(colorize('"' + basename + '"', "green", own) + " in " + func)
            print(
                "%s:  %s"
                % (
                    colorize("%5d" % line_num, "red", own),
                    colorize(text, "yellow", own),
                )
            )
            print("  %s" % colorize(filename, "black", own))

        print(colorize("%s: %s" % (type_.__name__, value), "cyan"))

    return hook
